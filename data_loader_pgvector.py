#!/usr/bin/env python3
"""
data_loader_pgvector.py  —  Fast parallel vector loader for pgvector tables
════════════════════════════════════════════════════════════════════════════
Identical to data_loader.py but targets pgvector tables (same schema,
different naming convention: bench_pgv_1m, bench_pgv_5m, bench_pgv_10m).

Uses COPY BINARY for maximum throughput (~30K vec/s with 4 workers).

Usage
─────
  python data_loader_pgvector.py \\
    --dsn "postgresql://enterprisedb:admin@localhost:5444/repsol" \\
    --target 1m --tablename bench_pgv_1m --workers 4

  python data_loader_pgvector.py --dsn "..." --target 5m  --tablename bench_pgv_5m
  python data_loader_pgvector.py --dsn "..." --target 10m --tablename bench_pgv_10m

  # Fall back to COPY TEXT if binary has issues:
  python data_loader_pgvector.py --dsn "..." --target 1m --tablename bench_pgv_1m --text

Requirements
────────────
  pip install psycopg2-binary pgvector numpy
"""

import argparse, io, math, multiprocessing as mp, struct, sys, time
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

SCALE_MAP = {
    "1m":  1_000_000,
    "5m":  5_000_000,
    "10m": 10_000_000,
    "50m": 50_000_000,
}
DEFAULT_TABLE   = "bench_pgv_1m"
DEFAULT_DIM     = 1536
DEFAULT_BATCH   = 10_000
DEFAULT_WORKERS = 4

G="\033[92m"; A="\033[93m"; T="\033[96m"; D="\033[2m"; B="\033[1m"; R="\033[0m"

_SOURCES  = ["internal", "external"]
_TAG_POOL = ["{ml,training}", "{infra}", "{security}", "{ml,research}", "{python}"]
_DEPTS    = ["engineering", "data-science", "platform", "security", "research"]

COPY_BINARY_SIG = b'PGCOPY\n\xff\r\n\x00' + struct.pack('>ii', 0, 0)
TEXT_OID = 25


def _pg_text(s: str) -> bytes:
    b = s.encode()
    return struct.pack('>i', len(b)) + b


def _pg_jsonb(s: str) -> bytes:
    b = b'\x01' + s.encode()
    return struct.pack('>i', len(b)) + b


def _pg_text_array(items) -> bytes:
    n = len(items)
    hdr   = struct.pack('>iiiii', 1, 0, TEXT_OID, n, 1)
    elems = b''.join(struct.pack('>i', len(s.encode())) + s.encode() for s in items)
    data  = hdr + elems
    return struct.pack('>i', len(data)) + data


def ensure_schema(dsn, table, dim):
    c = psycopg2.connect(dsn); c.autocommit = False
    register_vector(c); cur = c.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id         BIGSERIAL PRIMARY KEY,
            content    TEXT          NOT NULL,
            embedding  vector({dim}),
            source     TEXT,
            tags       TEXT[],
            meta       JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    c.commit(); cur.close(); c.close()
    print(f"  {G}✓{R}  Table {table!r} ready (dim={dim})")


def row_count(dsn, table):
    c = psycopg2.connect(dsn); cur = c.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    n = cur.fetchone()[0]; cur.close(); c.close()
    return n


def build_binary_buffer(offset, size, dim, rng):
    vecs  = rng.standard_normal((size, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-9)
    vec_hdr      = struct.pack('>HH', dim, 0)
    be_vecs      = vecs.astype('>f4')
    vec_field_len = struct.pack('>i', 4 + dim * 4)
    field_count  = struct.pack('>h', 5)

    buf = io.BytesIO()
    buf.write(COPY_BINARY_SIG)
    for i in range(size):
        gi         = offset + i
        content    = f"Chunk {gi:010d}"
        source     = _SOURCES[gi % 2]
        tag_items  = _TAG_POOL[gi % len(_TAG_POOL)].strip('{}').split(',')
        conf       = f"{0.55 + (gi % 45) / 100:.2f}"
        dept       = _DEPTS[gi % len(_DEPTS)]
        meta       = f'{{"confidence":{conf},"department":"{dept}"}}'
        vec_bytes  = vec_hdr + be_vecs[i].tobytes()
        buf.write(field_count)
        buf.write(_pg_text(content))
        buf.write(vec_field_len + vec_bytes)
        buf.write(_pg_text(source))
        buf.write(_pg_text_array(tag_items))
        buf.write(_pg_jsonb(meta))
    buf.write(struct.pack('>h', -1))
    buf.seek(0)
    return buf


def build_text_buffer(offset, size, dim, rng):
    vecs  = rng.standard_normal((size, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.maximum(norms, 1e-9)
    vec_buf = io.BytesIO()
    np.savetxt(vec_buf, vecs, delimiter=',', fmt='%.5g')
    vec_lines = vec_buf.getvalue().split(b'\n')
    buf = io.BytesIO()
    for i in range(size):
        if i >= len(vec_lines) or not vec_lines[i]: continue
        gi     = offset + i
        source = _SOURCES[gi % 2]
        tags   = _TAG_POOL[gi % len(_TAG_POOL)]
        conf   = f"{0.55 + (gi % 45) / 100:.2f}"
        dept   = _DEPTS[gi % len(_DEPTS)]
        meta   = f'{{"confidence":{conf},"department":"{dept}"}}'
        buf.write(f"Chunk {gi:010d}".encode() + b'\t[' + vec_lines[i] + b']\t' +
                  source.encode() + b'\t' + tags.encode() + b'\t' + meta.encode() + b'\n')
    buf.seek(0)
    return buf


def fmt_eta(s):
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

def draw_bar(pct, width=36):
    f = int(width * pct)
    return "█"*f + "░"*(width - f)


def _worker(worker_id, dsn, table, dim, batch, start_offset, count, use_binary, queue):
    try:
        conn = psycopg2.connect(dsn); conn.autocommit = False
        register_vector(conn); cur = conn.cursor()
        rng      = np.random.default_rng(start_offset ^ (worker_id * 0xC0FFEE42))
        done     = 0
        copy_cmd = (f"COPY {table} (content, embedding, source, tags, meta) "
                    f"FROM STDIN {'BINARY' if use_binary else ''}")
        build_fn = build_binary_buffer if use_binary else build_text_buffer
        while done < count:
            size = min(batch, count - done)
            buf  = build_fn(start_offset + done, size, dim, rng)
            cur.copy_expert(copy_cmd, buf)
            conn.commit()
            done += size
            queue.put(size)
        cur.close(); conn.close()
        queue.put(("done", worker_id))
    except Exception as e:
        queue.put(("error", worker_id, str(e)))


def load(dsn, table, target, dim, batch, n_workers, use_binary):
    ensure_schema(dsn, table, dim)
    current = row_count(dsn, table)
    needed  = target - current
    mode    = "COPY BINARY" if use_binary else "COPY TEXT"

    print(f"\n  Table   : {T}{table}{R}")
    print(f"  Target  : {target:>15,} vectors")
    print(f"  Current : {current:>15,} vectors")
    print(f"  To load : {needed:>15,} vectors")
    print(f"  Batch   : {batch:>15,}  ({mode})")
    print(f"  Workers : {n_workers:>15,}  (parallel processes)")
    print(f"  Dim     : {dim:>15,}")

    if needed <= 0:
        print(f"\n  {G}Already at or above target — nothing to do.{R}"); return

    chunk = math.ceil(needed / n_workers)
    queue = mp.Queue()
    procs = []
    for i in range(n_workers):
        w_start = current + i * chunk
        w_count = min(chunk, needed - i * chunk)
        if w_count <= 0: break
        p = mp.Process(target=_worker,
                       args=(i, dsn, table, dim, batch, w_start, w_count, use_binary, queue),
                       daemon=True)
        procs.append(p)

    print(f"\n  Starting {len(procs)} worker processes…\n")
    t0 = time.perf_counter()
    for p in procs: p.start()

    loaded = 0; finished = 0; errors = []
    while finished < len(procs):
        msg = queue.get()
        if isinstance(msg, int):
            loaded += msg
            elapsed = time.perf_counter() - t0
            rate    = loaded / elapsed if elapsed > 0 else 0
            eta     = (needed - loaded) / rate if rate > 0 and loaded < needed else 0
            sys.stdout.write(
                f"\r  [{draw_bar(loaded/needed)}] "
                f"{T}{current+loaded:>13,}{R}/{target:,}  "
                f"{A}{rate:>7,.0f}{R} vec/s  ETA {fmt_eta(eta)}   ")
            sys.stdout.flush()
        elif msg[0] == "done":  finished += 1
        elif msg[0] == "error": errors.append(f"Worker {msg[1]}: {msg[2]}"); finished += 1

    for p in procs: p.join()
    elapsed = time.perf_counter() - t0
    rate    = needed / elapsed if elapsed > 0 else 0

    if errors:
        print(f"\n\n  {A}Errors:{R}")
        for e in errors: print(f"    {e}")
        if use_binary: print(f"\n  {A}Hint:{R} try --text flag")
        return

    sys.stdout.write(
        f"\r  [{draw_bar(1.0)}] {T}{target:>13,}{R}/{target:,}  "
        f"{A}{rate:>7,.0f}{R} vec/s  done        \n")
    sys.stdout.flush()
    final = row_count(dsn, table)
    print(f"\n  {G}✓{R}  Loaded {needed:,} rows in {elapsed:.1f}s  ({rate:,.0f} vec/s avg)")
    print(f"  {G}✓{R}  Total rows in {table!r}: {final:,}")
    print(f"\n  Next → python build_index_pgvector.py --dsn '...' --tablename {table}")


def main():
    p = argparse.ArgumentParser(description="Fast parallel pgvector loader")
    p.add_argument("--dsn",       required=True)
    p.add_argument("--target",    choices=["1m","5m","10m","50m"], required=True)
    p.add_argument("--tablename", default=DEFAULT_TABLE)
    p.add_argument("--dim",       type=int, default=DEFAULT_DIM)
    p.add_argument("--batch",     type=int, default=DEFAULT_BATCH)
    p.add_argument("--workers",   type=int, default=DEFAULT_WORKERS)
    p.add_argument("--text",      action="store_true",
                   help="Use COPY TEXT instead of COPY BINARY")
    args = p.parse_args()

    target = SCALE_MAP[args.target]
    mode   = "COPY TEXT" if args.text else "COPY BINARY"
    print(f"\n{B}data_loader_pgvector.py — pgvector  [{mode}]{R}")
    print(f"  Scale     : {T}{args.target.upper()}{R} ({target:,} vectors)")
    print(f"  Table     : {T}{args.tablename}{R}")
    print(f"  Workers   : {T}{args.workers}{R}")
    print(f"  DSN       : {D}{args.dsn}{R}")
    load(args.dsn, args.tablename, target, args.dim, args.batch, args.workers, not args.text)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()


