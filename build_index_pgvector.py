#!/usr/bin/env python3
"""
build_index_pgvector.py  —  Build pgvector HNSW index
══════════════════════════════════════════════════════
Builds a pgvector HNSW index (NOT vchordrq) for comparison with VectorChord.
Use this on bench_pgv_* tables.

Key difference vs VectorChord:
  VectorChord:  CREATE INDEX ... USING vchordrq ... WITH (options = $$TOML$$)
                Query:  SET vchordrq.probes = 'N'
  pgvector:     CREATE INDEX ... USING hnsw   ... WITH (m=16, ef_construction=200)
                Query:  SET hnsw.ef_search = 100

Usage
─────
  python build_index_pgvector.py \\
    --dsn "postgresql://enterprisedb:admin@localhost:5444/repsol" \\
    --tablename bench_pgv_1m

  python build_index_pgvector.py --dsn "..." --tablename bench_pgv_5m  --m 16 --efc 200
  python build_index_pgvector.py --dsn "..." --tablename bench_pgv_10m --m 32 --efc 200

Recommended m / ef_construction by scale:
  1M   →  m=16  ef_construction=200   (standard)
  5M+  →  m=24  ef_construction=200   (better recall)
  10M+ →  m=32  ef_construction=200   (best recall, slower build)

Requirements
────────────
  pip install psycopg2-binary pgvector
"""

import argparse, sys, time, threading
import psycopg2

DEFAULT_TABLE = "bench_pgv_1m"

G="\033[92m"; A="\033[93m"; T="\033[96m"; B="\033[1m"; D="\033[2m"; R="\033[0m"


def idx_name(table):
    return f"{table}_hnsw"


def get_row_count(dsn, table):
    c = psycopg2.connect(dsn); cur = c.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = cur.fetchone()[0]; cur.close(); c.close()
    return n


def poll_progress(dsn, table, stop_evt):
    try:
        mon  = psycopg2.connect(dsn); mcur = mon.cursor()
        t0   = time.perf_counter()
        while not stop_evt.is_set():
            try:
                mcur.execute("""
                    SELECT phase,
                           COALESCE(blocks_done, 0),
                           COALESCE(blocks_total, 0),
                           COALESCE(tuples_done, 0),
                           COALESCE(tuples_total, 0)
                    FROM   pg_stat_progress_create_index
                    WHERE  relid = %s::regclass
                """, (table,))
                row = mcur.fetchone()
                if row:
                    phase, bd, bt, td, tt = row
                    pct  = f"{100*bd//bt}%" if bt > 0 else "—"
                    tups = f"tuples {td:,}/{tt:,}" if tt > 0 else ""
                    ela  = time.perf_counter() - t0
                    sys.stdout.write(
                        f"\r  [{phase}]  {pct:>5}  blocks {bd:,}/{bt:,}  "
                        f"{tups}  {ela:.0f}s   ")
                    sys.stdout.flush()
            except Exception:
                pass
            time.sleep(2)
        mcur.close(); mon.close()
    except Exception:
        pass


def build(dsn, table, m, efc, workers, mem, cosine):
    opclass  = "vector_cosine_ops" if cosine else "vector_l2_ops"
    operator = "<=>" if cosine else "<->"
    idx      = idx_name(table)

    print(f"  Table      : {T}{table}{R}")
    print(f"  Index      : {idx}")
    print(f"  Type       : pgvector HNSW")
    print(f"  m          : {m}")
    print(f"  ef_constr  : {efc}")
    print(f"  op class   : {opclass}  ({operator})")
    print(f"  workers    : {workers}")
    print()

    c = psycopg2.connect(dsn); c.autocommit = True; cur = c.cursor()
    cur.execute(f"DROP INDEX IF EXISTS {idx}")

    # Session settings to maximise parallel build
    cur.execute(f"SET max_parallel_maintenance_workers = {workers}")
    cur.execute(f"SET maintenance_work_mem = '{mem}'")
    print(f"  {T}Session:{R}  max_parallel_maintenance_workers={workers}  maintenance_work_mem={mem}\n")

    stop_evt = threading.Event()
    poller   = threading.Thread(target=poll_progress, args=(dsn, table, stop_evt), daemon=True)
    poller.start()

    t0  = time.perf_counter()
    err = [None]
    try:
        cur.execute(f"""
            CREATE INDEX {idx} ON {table}
            USING hnsw (embedding {opclass})
            WITH (m = {m}, ef_construction = {efc})
        """)
    except Exception as e:
        err[0] = e
    finally:
        stop_evt.set()

    elapsed = time.perf_counter() - t0
    cur.close(); c.close()

    if err[0]:
        print(f"\n\n  {A}BUILD FAILED: {err[0]}{R}")
        raise err[0]

    print(f"\n\n  {G}✓{R}  HNSW index built in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # Verify
    c2 = psycopg2.connect(dsn); cur2 = c2.cursor()
    cur2.execute("""
        SELECT pg_size_pretty(pg_relation_size(ix.indexrelid)), ix.indisvalid
        FROM   pg_index ix
        JOIN   pg_class i ON i.oid = ix.indexrelid
        WHERE  i.relname = %s
    """, (idx,))
    row = cur2.fetchone()
    if row:
        print(f"  {G}✓{R}  Index size: {row[0]}   valid: {row[1]}")
    cur2.close(); c2.close()

    print(f"\n  {T}Query setting:{R}  SET hnsw.ef_search = 100;")
    print(f"  Higher ef_search = better recall, lower QPS")
    return elapsed


def main():
    p = argparse.ArgumentParser(description="Build pgvector HNSW index")
    p.add_argument("--dsn",       required=True)
    p.add_argument("--tablename", default=DEFAULT_TABLE)
    p.add_argument("--m",         type=int, default=16,
                   help="HNSW m — graph connectivity (default 16)")
    p.add_argument("--efc",       type=int, default=200,
                   help="ef_construction — build quality (default 200)")
    p.add_argument("--workers",   type=int, default=8,
                   help="max_parallel_maintenance_workers (default 8)")
    p.add_argument("--mem",       default="4GB",
                   help="maintenance_work_mem (default 4GB)")
    p.add_argument("--l2",        action="store_true",
                   help="Use L2 distance instead of cosine")
    args = p.parse_args()

    print(f"\n{B}build_index_pgvector.py — pgvector HNSW{R}")
    print(f"  DSN: {D}{args.dsn}{R}\n")

    n = get_row_count(args.dsn, args.tablename)
    print(f"  Rows in {T}{args.tablename}{R}: {n:,}")

    # Recommend m based on scale
    if n >= 10_000_000:
        rec_m = 32
    elif n >= 5_000_000:
        rec_m = 24
    else:
        rec_m = 16
    if args.m == 16 and rec_m != 16:
        print(f"  {A}Note:{R} for {n:,} rows, consider --m {rec_m} for better recall")

    build(args.dsn, args.tablename, args.m, args.efc,
          args.workers, args.mem, cosine=not args.l2)


if __name__ == "__main__":
    main()



