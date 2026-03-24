#!/usr/bin/env python3
"""
build_index.py  —  Build VectorChord vchordrq index
════════════════════════════════════════════════════
Run after data_loader.py. Monitors progress live.

Usage
─────
  # Default table:
  python build_index.py --dsn "postgresql://enterprisedb:admin@localhost:5444/repsol"

  # Dedicated per-scale tables:
  python build_index.py --dsn "..." --tablename scale_bench_docs_1m  --lists 2000
  python build_index.py --dsn "..." --tablename scale_bench_docs_5m  --lists 4000
  python build_index.py --dsn "..." --tablename scale_bench_docs_10m --lists 8000
  python build_index.py --dsn "..." --tablename scale_bench_docs_50m --lists 40000

  # Auto-select lists from row count (recommended):
  python build_index.py --dsn "..." --tablename scale_bench_docs_1m

Recommended lists by row count:
  < 500K   →  lists = []       (no partitioning)
    1M     →  lists = [2000]
    5M     →  lists = [4000]
   10M     →  lists = [8000]
   50M     →  lists = [40000]

Requirements
────────────
  pip install psycopg2-binary
"""

import argparse, sys, time, threading
import psycopg2

DEFAULT_TABLE = "scale_bench_docs"

G="\033[92m"; A="\033[93m"; T="\033[96m"; B="\033[1m"; D="\033[2m"; R="\033[0m"


def get_row_count(dsn, table):
    c = psycopg2.connect(dsn); cur = c.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = cur.fetchone()[0]; cur.close(); c.close()
    return n


def recommend_lists(n):
    if n < 500_000:    return []
    if n < 2_000_000:  return [2000]
    if n < 7_000_000:  return [4000]
    if n < 20_000_000: return [8000]
    return [40000]


def idx_name(table):
    """Derive index name from table name."""
    return f"{table}_vchordrq"


def poll_progress(dsn, table, stop_evt):
    try:
        mon = psycopg2.connect(dsn); mcur = mon.cursor()
        t0  = time.perf_counter()
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
                    pct   = f"{100*bd//bt}%" if bt > 0 else "—"
                    tups  = f"tuples {td:,}/{tt:,}" if tt > 0 else ""
                    ela   = time.perf_counter() - t0
                    sys.stdout.write(
                        f"\r  [{phase}]  {pct:>5}  blocks {bd:,}/{bt:,}  "
                        f"{tups}  {ela:.0f}s   "
                    )
                    sys.stdout.flush()
            except Exception:
                pass
            time.sleep(2)
        mcur.close(); mon.close()
    except Exception:
        pass


def build(dsn, table, lists, threads, cosine, pin, residual, mem="4GB"):
    opclass   = "vector_cosine_ops" if cosine else "vector_l2_ops"
    spherical = "true" if cosine else "false"
    lists_str = str(lists).replace(' ', '')
    idx       = idx_name(table)

    toml = f"""residual_quantization = {'true' if residual else 'false'}
build.pin = {pin}
[build.internal]
lists = {lists_str}
spherical_centroids = {spherical}
build_threads = {threads}
"""
    print(f"  Table      : {T}{table}{R}")
    print(f"  Index      : {idx}")
    print(f"  lists      : {lists_str}")
    print(f"  op class   : {opclass}")
    print(f"  threads    : {threads}")
    print(f"  residual   : {residual}")
    print(f"  pin        : {pin}")
    print()

    c = psycopg2.connect(dsn); c.autocommit = True; cur = c.cursor()
    cur.execute(f"DROP INDEX IF EXISTS {idx}")

    # Maximise parallelism for this session before building
    cur.execute("SET max_parallel_maintenance_workers = 8")
    cur.execute("SET max_parallel_workers = 8")
    cur.execute(f"SET maintenance_work_mem = '{mem}'")
    print(f"  {T}Session settings:{R} max_parallel_maintenance_workers=8  maintenance_work_mem={mem}")
    print(f"  Building index…\n")

    stop_evt = threading.Event()
    poller   = threading.Thread(target=poll_progress, args=(dsn, table, stop_evt), daemon=True)
    poller.start()

    t0  = time.perf_counter()
    err = [None]
    try:
        cur.execute(f"""
            CREATE INDEX {idx} ON {table}
            USING vchordrq (embedding {opclass})
            WITH (options = $opt${toml}$opt$)
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

    print(f"\n\n  {G}✓{R}  Index built in {elapsed:.1f}s  ({elapsed/60:.1f} min)")

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

    probes_hint = max(1, lists[0] // 200) if lists else 1
    print(f"\n  {T}Query setting:{R}  SET vchordrq.probes = '{probes_hint}';")
    print(f"  {D}Switch to this table in the app header dropdown.{R}")
    return elapsed


def main():
    p = argparse.ArgumentParser(description="Build VectorChord vchordrq index")
    p.add_argument("--dsn",         required=True)
    p.add_argument("--tablename",   default=DEFAULT_TABLE,
                   help=f"Table to index (default: {DEFAULT_TABLE})")
    p.add_argument("--lists",       type=int, default=None,
                   help="Partition count — auto-chosen from row count if omitted")
    p.add_argument("--threads",     type=int, default=16,
                   help="vchordrq build_threads for K-means (default 16)")
    p.add_argument("--l2",          action="store_true",
                   help="Use L2 distance instead of cosine (default: cosine)")
    p.add_argument("--pin",         type=int, default=2)
    p.add_argument("--mem",         default="4GB",
                   help="maintenance_work_mem for this session (default 4GB)")
    p.add_argument("--no-residual", action="store_true")
    args = p.parse_args()

    print(f"\n{B}build_index.py — VectorChord vchordrq{R}")
    print(f"  DSN: {D}{args.dsn}{R}\n")

    n = get_row_count(args.dsn, args.tablename)
    print(f"  Rows in {T}{args.tablename}{R}: {n:,}")

    lists = [args.lists] if args.lists else recommend_lists(n)
    print(f"  lists = {lists}  (auto-recommended for {n:,} rows)")

    build(args.dsn, args.tablename, lists, args.threads,
          cosine=not args.l2, pin=args.pin, residual=not args.no_residual,
          mem=args.mem)


if __name__ == "__main__":
    main()


