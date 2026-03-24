#!/usr/bin/env python3
"""
VectorChord + EDB — Live Demo Server
Usage:
  pip install flask psycopg2-binary pgvector numpy
  python app.py --dsn "postgresql://enterprisedb:admin@localhost:5444/repsol"
  open http://localhost:5050
"""
import argparse, json, statistics, time, threading
import numpy as np
import psycopg2
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)
DSN = None
DIM = 1536
TABLE      = "scale_bench_docs"  # active table, switchable at runtime
INDEX_TYPE = "vchordrq"            # "vchordrq" or "hnsw" — set when table switches
TRACE_LOG  = []                    # in-memory trace of all benchmark runs
TRACE_MAX  = 10_000                # cap to avoid unbounded memory


def trace(run_type, **kwargs):
    """Append a trace entry. Called from every benchmark route."""
    import datetime
    entry = {
        "ts":         datetime.datetime.utcnow().isoformat() + "Z",
        "run_type":   run_type,
        "table":      TABLE,
        "index_type": INDEX_TYPE,
        **kwargs
    }
    TRACE_LOG.append(entry)
    if len(TRACE_LOG) > TRACE_MAX:
        TRACE_LOG.pop(0)

def get_conn():
    c = psycopg2.connect(DSN)
    c.autocommit = True
    return c


def _set(cur, sql):
    """Execute a SET statement and drain the cursor to avoid 'no results to fetch'."""
    cur.execute(sql)
    try:
        cur.fetchall()
    except Exception:
        pass  # SET statements have no result rows — safe to ignore


def set_search_param(cur, value):
    """Set the right search tuning parameter depending on active index type.
    Also disables seqscan so the planner always uses the vector index."""
    _set(cur, "SET enable_seqscan = off")
    if INDEX_TYPE == "hnsw":
        _set(cur, f"SET hnsw.ef_search = {value}")
    elif INDEX_TYPE == "ivfflat":
        _set(cur, f"SET ivfflat.probes = {value}")
    else:
        _set(cur, f"SET vchordrq.probes = '{value}'")


def ensure_analyzed(cur, tbl):
    """Run ANALYZE if table has never been analyzed (stale planner stats = seqscan)."""
    cur.execute("""
        SELECT last_analyze, last_autoanalyze
        FROM   pg_stat_user_tables WHERE relname = %s
    """, (tbl,))
    row = cur.fetchone()
    if row and row[0] is None and row[1] is None:
        cur.execute(f"ANALYZE {tbl}")


def vec_str(v):
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"

def rand_vec(seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v

def pct(data, p):
    return round(data[max(0, int(len(data)*p/100)-1)], 2)


@app.route("/api/status")
def status():
    tbl = TABLE
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','vchord')")
        exts = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        # Ensure planner stats are fresh so index is used
        cur.execute(f"ANALYZE {tbl}")
        rows = cur.fetchone()[0]
        cur.execute(f"SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) FROM pg_indexes WHERE tablename='{tbl}' AND indexname != '{tbl}_pkey'")
        indexes = [{"name": r[0], "size": r[1]} for r in cur.fetchall()]
        cur.close(); c.close()
        return jsonify({"ok": True, "version": ver, "extensions": exts, "rows": rows, "indexes": indexes, "index_type": INDEX_TYPE, "table": TABLE})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/tables")
def list_tables():
    """Discover all tables that have a vector column — lets user switch source table."""
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("""
            SELECT t.relname                                         AS table_name,
                   c.attname                                         AS vec_column,
                   tp.typname                                        AS vec_type,
                   pg_size_pretty(pg_total_relation_size(t.relname::regclass)) AS size
            FROM   pg_attribute  c
            JOIN   pg_class      t  ON t.oid = c.attrelid
            JOIN   pg_type       tp ON tp.oid = c.atttypid
            JOIN   pg_namespace  ns ON ns.oid = t.relnamespace
            WHERE  tp.typname IN ('vector','halfvec')
              AND  t.relkind = 'r'
              AND  ns.nspname = 'public'
              AND  c.attnum > 0
            ORDER  BY t.relname
        """)
        rows = cur.fetchall()
        tables = []
        for r in rows:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {r[0]}")
                cnt = cur.fetchone()[0]
            except:
                cnt = 0
            # Detect index type: vchordrq or hnsw
            cur.execute("""
                SELECT am.amname, i.relname
                FROM   pg_index ix
                JOIN   pg_class i  ON i.oid  = ix.indexrelid
                JOIN   pg_class t  ON t.oid  = ix.indrelid
                JOIN   pg_am    am ON am.oid = i.relam
                WHERE  t.relname = %s
                  AND  am.amname IN ('vchordrq','hnsw','ivfflat')
                LIMIT  1
            """, (r[0],))
            idx_row = cur.fetchone()
            idx_type = idx_row[0] if idx_row else "none"
            idx_name = idx_row[1] if idx_row else None
            tables.append({
                "name": r[0], "vec_col": r[1], "vec_type": r[2],
                "size": r[3], "rows": cnt,
                "index_type": idx_type, "index_name": idx_name
            })
        cur.close(); c.close()
        return jsonify({"tables": tables, "active": TABLE})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trace")
def get_trace():
    """Return all recorded trace entries, optionally filtered."""
    run_type = request.args.get("type")       # filter by run_type
    tbl      = request.args.get("table")      # filter by table
    fmt      = request.args.get("fmt", "json")# json or csv
    entries  = TRACE_LOG[:]
    if run_type: entries = [e for e in entries if e.get("run_type") == run_type]
    if tbl:      entries = [e for e in entries if e.get("table") == tbl]
    if fmt == "csv":
        import csv, io
        if not entries:
            return "ts,run_type,table,index_type,p50,p95,p99,qps,latency_ms,threads,mode,metric\n", 200, {"Content-Type":"text/csv","Content-Disposition":"attachment; filename=trace.csv"}
        keys = list(dict.fromkeys(k for e in entries for k in e))
        buf  = io.StringIO()
        w    = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(entries)
        return buf.getvalue(), 200, {
            "Content-Type":        "text/csv",
            "Content-Disposition": "attachment; filename=trace.csv"
        }
    return jsonify({"count": len(entries), "entries": entries})


@app.route("/api/trace", methods=["DELETE"])
def clear_trace():
    TRACE_LOG.clear()
    return jsonify({"ok": True, "message": "Trace log cleared"})


@app.route("/api/explain")
def explain():
    """Run EXPLAIN (ANALYZE, BUFFERS) on a single vector query and return the plan."""
    global TABLE, INDEX_TYPE
    tbl    = TABLE
    probes = request.args.get("probes", 10)
    topk   = int(request.args.get("topk", 10))
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("SET enable_seqscan = off")
        set_search_param(cur, probes)

        q       = rand_vec(42)
        vs      = vec_str(q)
        sql     = f"""EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, content, embedding <=> %s::vector AS dist
FROM   {tbl}
ORDER  BY embedding <=> %s::vector
LIMIT  {topk}"""

        cur.execute(sql, (vs, vs))
        rows  = cur.fetchall()
        plan  = "\n".join(r[0] for r in rows)

        # Also get the raw EXPLAIN (no ANALYZE) for cost estimates
        cur.execute(
            f"EXPLAIN (FORMAT TEXT) SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
            (vs, topk)
        )
        cost_plan = "\n".join(r[0] for r in cur.fetchall())

        # Detect if index was used
        index_used  = any(x in plan for x in ["Index Scan", "vchordrq", "hnsw", "ivfflat"])
        seq_scan    = "Seq Scan" in plan or "Parallel Seq Scan" in plan

        cur.close(); c.close()

        trace("explain", probes=probes, topk=topk,
              index_used=index_used, seq_scan=seq_scan)

        return jsonify({
            "table":       tbl,
            "index_type":  INDEX_TYPE,
            "probes":      probes,
            "topk":        topk,
            "plan":        plan,
            "cost_plan":   cost_plan,
            "index_used":  index_used,
            "seq_scan":    seq_scan,
            "warning":     "Index NOT used — check ANALYZE and enable_seqscan" if seq_scan else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug_tags")
def debug_tags():
    """Show sample tag values and test filter — for debugging metadata section."""
    global TABLE
    tbl = TABLE
    try:
        c = get_conn(); cur = c.cursor()
        # Show distinct tag arrays stored
        cur.execute(f"""
            SELECT tags, COUNT(*) as cnt
            FROM   {tbl}
            GROUP  BY tags
            ORDER  BY cnt DESC
            LIMIT  20
        """)
        stored = [{"tags": list(r[0]) if r[0] else [], "count": r[1]} for r in cur.fetchall()]
        # Test each chip value
        chips = ['ml', 'infra', 'security', 'training', 'research', 'python']
        chip_counts = {}
        for chip in chips:
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE tags @> ARRAY[%s]", (chip,))
            chip_counts[chip] = cur.fetchone()[0]
        cur.close(); c.close()
        return jsonify({"table": tbl, "stored_tags": stored, "chip_counts": chip_counts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def run_query():
    """Execute arbitrary SQL and return results — for the query editor."""
    global TABLE, INDEX_TYPE
    data    = request.get_json()
    sql     = data.get("sql", "").strip()
    timeout = int(data.get("timeout_ms", 10000))

    if not sql:
        return jsonify({"error": "No SQL provided"}), 400

    # Safety: only allow SELECT / SET / SHOW / EXPLAIN
    first_word = sql.lstrip().split()[0].upper() if sql.strip() else ""
    if first_word not in ("SELECT", "SET", "SHOW", "EXPLAIN", "ANALYZE", "WITH"):
        return jsonify({"error": f"Only SELECT / SET / SHOW / EXPLAIN / WITH allowed (got {first_word})"}), 400

    try:
        c = get_conn(); cur = c.cursor()
        cur.execute(f"SET statement_timeout = {timeout}")
        t0 = time.perf_counter()
        cur.execute(sql)
        elapsed = (time.perf_counter() - t0) * 1000

        # Collect results
        try:
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(200)   # cap at 200 rows for display
            total = len(rows)
        except Exception:
            cols, rows, total = [], [], 0

        cur.close(); c.close()
        trace("query_editor", latency_ms=round(elapsed, 2),
              sql_preview=sql[:120], rows_returned=total)
        return jsonify({
            "ok":          True,
            "latency_ms":  round(elapsed, 2),
            "columns":     cols,
            "rows":        [[str(v)[:200] for v in r] for r in rows],
            "row_count":   total,
            "truncated":   total == 200,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/full_query")
def full_query():
    """Return a complete, executable SQL query with a real vector substituted in."""
    global TABLE, INDEX_TYPE
    tbl    = TABLE
    probes = request.args.get("probes", 10)
    topk   = int(request.args.get("topk", 10))
    mode   = request.args.get("mode", "vector")   # vector | hybrid | metadata

    q  = rand_vec(42)
    vs = vec_str(q)

    if INDEX_TYPE == "hnsw":
        set_stmt = f"SET hnsw.ef_search = {probes};"
    else:
        set_stmt = f"SET vchordrq.probes = \'{probes}\';"

    if mode == "hybrid":
        sql = f"""{set_stmt}

WITH vec AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> \'{vs}\'::vector) AS rnk
    FROM   {tbl}
    LIMIT  100
),
kw AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(to_tsvector(\'english\', content),
                                to_tsquery(\'english\', \'machine & learning\')) DESC
           ) AS rnk
    FROM   {tbl}
    WHERE  to_tsvector(\'english\', content)
             @@ to_tsquery(\'english\', \'machine & learning\')
    LIMIT  100
)
SELECT d.id,
       d.content,
       d.source,
       (COALESCE(1.0/(60+v.rnk),0) + COALESCE(1.0/(60+k.rnk),0)) AS rrf_score
FROM   {tbl} d
LEFT   JOIN vec v USING (id)
LEFT   JOIN kw  k USING (id)
WHERE  v.id IS NOT NULL OR k.id IS NOT NULL
ORDER  BY rrf_score DESC
LIMIT  {topk};"""
    elif mode == "metadata":
        sql = f"""{set_stmt}

SELECT id,
       content,
       source,
       tags,
       (meta->>\'confidence\')::float AS confidence,
       embedding <=> \'{vs}\'::vector  AS dist
FROM   {tbl}
WHERE  source = \'internal\'
  AND  \'ml\' = ANY(tags)
ORDER  BY embedding <=> \'{vs}\'::vector
LIMIT  {topk};"""
    else:
        sql = f"""{set_stmt}

SELECT id,
       content,
       source,
       tags,
       embedding <=> \'{vs}\'::vector AS dist
FROM   {tbl}
ORDER  BY embedding <=> \'{vs}\'::vector
LIMIT  {topk};"""

    return jsonify({
        "table":      tbl,
        "index_type": INDEX_TYPE,
        "probes":     probes,
        "topk":       topk,
        "mode":       mode,
        "sql":        sql,
        "vector_dim": DIM,
        "vector_preview": vs[:80] + "..." if len(vs) > 80 else vs,
    })


@app.route("/api/recall")
def recall():
    """
    Compute Recall@K by comparing ANN results vs exact brute-force (sequential scan).
    Recall@K = |ANN_ids ∩ exact_ids| / K
    Runs `samples` queries, returns mean recall.
    NOTE: exact scan on large tables is slow — keep samples low (5-10).
    """
    global TABLE, INDEX_TYPE
    tbl     = TABLE
    topk    = int(request.args.get("topk", 10))
    probes  = int(request.args.get("probes", 10))
    samples = int(request.args.get("samples", 10))
    try:
        c = get_conn(); cur = c.cursor()

        recalls = []
        ann_times = []
        exact_times = []

        for i in range(samples):
            q  = rand_vec(i + 9000)   # different seeds from latency benchmark
            vs = vec_str(q)

            # ANN query (uses index)
            set_search_param(cur, probes)
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vs, topk))
            ann_ids = {r[0] for r in cur.fetchall()}
            ann_times.append((time.perf_counter() - t0) * 1000)

            # Exact query (force sequential scan)
            _set(cur, "SET enable_seqscan = on")
            _set(cur, "SET enable_indexscan = off")
            _set(cur, "SET enable_bitmapscan = off")
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vs, topk))
            exact_ids = {r[0] for r in cur.fetchall()}
            exact_times.append((time.perf_counter() - t0) * 1000)

            # Restore index usage
            _set(cur, "SET enable_seqscan = off")
            _set(cur, "SET enable_indexscan = on")
            _set(cur, "SET enable_bitmapscan = on")

            if exact_ids:
                recalls.append(len(ann_ids & exact_ids) / len(exact_ids))

        cur.close(); c.close()

        mean_recall  = round(sum(recalls) / len(recalls) * 100, 1) if recalls else 0
        mean_ann_ms  = round(sum(ann_times) / len(ann_times), 2)
        mean_exact_ms = round(sum(exact_times) / len(exact_times), 2)

        trace("recall", probes=probes, topk=topk, samples=samples,
              recall_pct=mean_recall, ann_ms=mean_ann_ms)

        return jsonify({
            "table":        tbl,
            "index_type":   INDEX_TYPE,
            "probes":       probes,
            "topk":         topk,
            "samples":      samples,
            "recall_pct":   mean_recall,
            "ann_ms":       mean_ann_ms,
            "exact_ms":     mean_exact_ms,
            "per_query":    [round(r*100,1) for r in recalls],
            "note": f"Recall@{topk}: {mean_recall}% — {samples} sample queries, probes={probes}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tuning")
def get_tuning():
    """Return current tuning parameters for the active index type."""
    try:
        c = get_conn(); cur = c.cursor()
        params = {}
        if INDEX_TYPE == "hnsw":
            cur.execute("SHOW hnsw.ef_search")
            params["hnsw.ef_search"] = cur.fetchone()[0]
            cur.execute("SHOW hnsw.iterative_scan")
            params["hnsw.iterative_scan"] = cur.fetchone()[0]
            # Index definition (m, ef_construction)
            tbl = TABLE
            cur.execute("""
                SELECT pg_get_indexdef(ix.indexrelid)
                FROM   pg_index ix
                JOIN   pg_class i ON i.oid = ix.indexrelid
                JOIN   pg_class t ON t.oid = ix.indrelid
                JOIN   pg_am   am ON am.oid = i.relam
                WHERE  t.relname = %s AND am.amname = 'hnsw'
                LIMIT 1
            """, (tbl,))
            row = cur.fetchone()
            params["index_def"] = row[0] if row else None
        else:  # vchordrq
            cur.execute("SHOW vchordrq.probes")
            params["vchordrq.probes"] = cur.fetchone()[0]
            cur.execute("SHOW vchordrq.epsilon")
            params["vchordrq.epsilon"] = cur.fetchone()[0]
            try:
                cur.execute("SHOW vchordrq.max_scan_tuples")
                params["vchordrq.max_scan_tuples"] = cur.fetchone()[0]
            except Exception:
                params["vchordrq.max_scan_tuples"] = "-1"
            # Index definition (lists, residual_quantization etc)
            tbl = TABLE
            cur.execute("""
                SELECT pg_get_indexdef(ix.indexrelid)
                FROM   pg_index ix
                JOIN   pg_class i ON i.oid = ix.indexrelid
                JOIN   pg_class t ON t.oid = ix.indrelid
                JOIN   pg_am   am ON am.oid = i.relam
                WHERE  t.relname = %s AND am.amname = 'vchordrq'
                LIMIT 1
            """, (tbl,))
            row = cur.fetchone()
            params["index_def"] = row[0] if row else None
        cur.close(); c.close()
        return jsonify({"ok": True, "index_type": INDEX_TYPE, "table": TABLE, "params": params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tuning", methods=["POST"])
def set_tuning():
    """Apply a tuning parameter for the current session — affects all subsequent queries."""
    global INDEX_TYPE
    data  = request.get_json()
    param = data.get("param", "").strip()
    value = str(data.get("value", "")).strip()

    # Whitelist — only allow known safe params
    ALLOWED = {
        "vchordrq.probes", "vchordrq.epsilon", "vchordrq.max_scan_tuples",
        "hnsw.ef_search", "hnsw.iterative_scan",
        "ivfflat.probes",
        "enable_seqscan",
        "max_parallel_workers_per_gather",
    }
    if param not in ALLOWED:
        return jsonify({"error": f"Parameter {param!r} not in allowed list"}), 400

    try:
        c = get_conn(); cur = c.cursor()
        # For vchordrq.probes, value must be quoted
        if param == "vchordrq.probes":
            cur.execute(f"SET {param} = '{value}'")
        else:
            cur.execute(f"SET {param} = {value}")
        # Read back to confirm
        cur.execute(f"SHOW {param}")
        actual = cur.fetchone()[0]
        cur.close(); c.close()
        return jsonify({"ok": True, "param": param, "value": actual})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/set_table", methods=["POST"])
def set_table():
    global TABLE
    data = request.get_json()
    name = data.get("table", "").strip()
    if not name or not name.replace('_','').replace('-','').isalnum():
        return jsonify({"error": "Invalid table name"}), 400
    # Verify it exists
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("SELECT COUNT(*) FROM pg_class WHERE relname = %s AND relkind = 'r'", (name,))
        if cur.fetchone()[0] == 0:
            return jsonify({"error": f"Table '{name}' not found"}), 404
        cur.execute(f"SELECT COUNT(*) FROM {name}")
        rows = cur.fetchone()[0]
        # Detect index type BEFORE closing cursor
        cur.execute("""
            SELECT am.amname FROM pg_index ix
            JOIN pg_class i  ON i.oid = ix.indexrelid
            JOIN pg_class t  ON t.oid = ix.indrelid
            JOIN pg_am    am ON am.oid = i.relam
            WHERE t.relname = %s AND am.amname IN ('vchordrq','hnsw','ivfflat')
            LIMIT 1
        """, (name,))
        irow = cur.fetchone()
        idx_type = irow[0] if irow else "vchordrq"
        cur.close(); c.close()
        TABLE = name
        INDEX_TYPE = idx_type
        return jsonify({"ok": True, "table": TABLE, "rows": rows, "index_type": INDEX_TYPE})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/latency")
def latency():
    global TABLE
    tbl = TABLE
    probes = int(request.args.get("probes", 10))
    topk   = int(request.args.get("topk", 10))
    iters  = int(request.args.get("iters", 50))
    SQL_SETUP = (f"SET hnsw.ef_search = {probes};" if INDEX_TYPE == "hnsw" else f"SET vchordrq.probes = '{probes}';")
    SQL_QUERY = f"SELECT id FROM {tbl}\nORDER BY embedding <=> $1::vector\nLIMIT {topk};"
    try:
        c = get_conn(); cur = c.cursor()
        set_search_param(cur, probes)
        for _ in range(5):
            q = rand_vec(42)
            cur.execute(f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str(q), topk))
            cur.fetchall()
        lats = []
        for i in range(iters):
            qi = rand_vec(i)
            t0 = time.perf_counter()
            cur.execute(f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s", (vec_str(qi), topk))
            rows = cur.fetchall()
            lats.append((time.perf_counter() - t0) * 1000)
        lats.sort()
        mean = statistics.mean(lats)
        cur.close(); c.close()
        # Build one fully-executable example query with a real vector
        sample_vec = vec_str(rand_vec(0))
        full_sql = (f"SET {'hnsw.ef_search' if INDEX_TYPE == 'hnsw' else 'vchordrq.probes'} = '{probes}';\n\n"
                    f"SELECT id, content,\n"
                    f"       embedding <=> '{sample_vec}'::vector AS dist\n"
                    f"FROM   {tbl}\n"
                    f"ORDER  BY embedding <=> '{sample_vec}'::vector\n"
                    f"LIMIT  {topk};")
        result = {
            "probes": probes, "topk": topk, "iters": iters,
            "p50": pct(lats,50), "p95": pct(lats,95), "p99": pct(lats,99),
            "mean": round(mean,2), "qps": round(1000/mean,1),
            "sql_setup": SQL_SETUP, "sql_query": SQL_QUERY,
            "full_sql": full_sql,
        }
        trace("latency", probes=probes, topk=topk, iters=iters,
              p50=result["p50"], p95=result["p95"], p99=result["p99"],
              mean=result["mean"], qps=result["qps"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/index_info")
def index_info():
    global TABLE
    tbl = TABLE
    SQL = f"""SELECT i.relname, am.amname,
       pg_size_pretty(pg_relation_size(ix.indexrelid)),
       ix.indisvalid
FROM   pg_index ix
JOIN   pg_class i  ON i.oid = ix.indexrelid
JOIN   pg_class t  ON t.oid = ix.indrelid
JOIN   pg_am    am ON am.oid = i.relam
WHERE  t.relname = '{tbl}'
  AND  i.relname != '{tbl}_pkey';"""
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute(SQL)
        indexes = [{"name":r[0],"type":r[1],"size":r[2],"valid":r[3]} for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        rows = cur.fetchone()[0]
        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{tbl}'))")
        table_size = cur.fetchone()[0]
        cur.close(); c.close()
        return jsonify({"indexes": indexes, "rows": rows, "table_size": table_size, "sql": SQL})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/hybrid")
def hybrid():
    global TABLE
    tbl = TABLE  # local copy — avoids f-string scoping issues
    query_text = request.args.get("q", "machine learning training")
    mode       = request.args.get("mode", "hybrid")
    topk       = int(request.args.get("topk", 10))
    probes     = int(request.args.get("probes", 10))
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("SET statement_timeout = '8000'")
        set_search_param(cur, probes)
        q = rand_vec(hash(query_text) % 10000)
        t0 = time.perf_counter()

        if mode == "vector":
            SQL = f"""-- Vector-only search (vchordrq index, probes={probes})
SET vchordrq.probes = '{probes}';

SELECT id, content, source, tags,
       embedding <=> $1::vector AS score
FROM   {tbl}
ORDER  BY embedding <=> $1::vector
LIMIT  {topk};"""
            cur.execute(f"SELECT id, content, source, tags, embedding <=> %s::vector AS score FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
                        (vec_str(q), vec_str(q), topk))

        elif mode == "keyword":
            words = [w for w in query_text.lower().split() if len(w) > 2]
            tsq   = " & ".join(words) if words else "data"
            SQL = f"""-- Full-text keyword search (GIN index on tsvector)
SELECT id, content, source, tags,
       ts_rank(to_tsvector('english', content),
               to_tsquery('english', '{tsq}')) AS score
FROM   {tbl}
WHERE  to_tsvector('english', content)
         @@ to_tsquery('english', '{tsq}')
ORDER  BY score DESC
LIMIT  {topk};"""
            cur.execute(f"SELECT id, content, source, tags, ts_rank(to_tsvector('english', content), to_tsquery('english', %s)) AS score FROM {tbl} WHERE to_tsvector('english', content) @@ to_tsquery('english', %s) ORDER BY score DESC LIMIT %s",
                        (tsq, tsq, topk))

        else:
            words = [w for w in query_text.lower().split() if len(w) > 2]
            tsq   = " & ".join(words) if words else "data"
            SQL = f"""-- Hybrid search: vector + keyword fused with RRF
-- probes={probes}, tsquery='{tsq}'
SET vchordrq.probes = '{probes}';

WITH vec AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk
    FROM   {tbl} LIMIT 100
),
kw AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(to_tsvector('english', content),
                                to_tsquery('english', '{tsq}')) DESC
           ) AS rnk
    FROM   {tbl}
    WHERE  to_tsvector('english', content)
             @@ to_tsquery('english', '{tsq}')
    LIMIT  100
)
SELECT d.id, d.content, d.source, d.tags,
       (COALESCE(1.0/(60+v.rnk),0) +
        COALESCE(1.0/(60+k.rnk),0)) AS rrf_score
FROM   {tbl} d
LEFT   JOIN vec v USING (id)
LEFT   JOIN kw  k USING (id)
WHERE  v.id IS NOT NULL OR k.id IS NOT NULL
ORDER  BY rrf_score DESC
LIMIT  {topk};"""
            cur.execute(f"""
                WITH vec AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rnk
                    FROM {tbl} LIMIT 100
                ),
                kw AS (
                    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', content), to_tsquery('english', %s)) DESC) AS rnk
                    FROM {tbl} WHERE to_tsvector('english', content) @@ to_tsquery('english', %s) LIMIT 100
                )
                SELECT d.id, d.content, d.source, d.tags,
                       (COALESCE(1.0/(60+v.rnk),0) + COALESCE(1.0/(60+k.rnk),0)) AS rrf_score
                FROM {tbl} d
                LEFT JOIN vec v USING (id) LEFT JOIN kw k USING (id)
                WHERE v.id IS NOT NULL OR k.id IS NOT NULL
                ORDER BY rrf_score DESC LIMIT %s""",
                (vec_str(q), tsq, tsq, topk))

        rows = cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        cur.close(); c.close()
        trace("hybrid", mode=mode, query=query_text,
              latency_ms=round(elapsed,2), result_count=len(rows))
        return jsonify({
            "mode": mode, "query": query_text, "latency_ms": round(elapsed, 2),
            "sql": SQL,
            "results": [{"id":r[0],"content":r[1][:120],"source":r[2],"tags":r[3],"score":round(float(r[4]),6)} for r in rows]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metadata")
def metadata():
    global TABLE
    tbl      = TABLE
    source   = request.args.get("source", "")
    tag      = request.args.get("tag", "")
    min_conf = float(request.args.get("min_conf", 0))
    topk     = int(request.args.get("topk", 10))
    probes   = int(request.args.get("probes", 10))
    try:
        c = get_conn(); cur = c.cursor()
        set_search_param(cur, probes)
        q  = rand_vec(99)
        vs = vec_str(q)

        # Build WHERE clause — each condition appended separately so param order is clear
        where_parts = []
        where_params = []          # params for the WHERE clause only
        if source:
            where_parts.append("source = %s")
            where_params.append(source)
        if tag:
            # Match either as array element OR substring (handles both storage formats)
            where_parts.append("(tags @> ARRAY[%s]::text[] OR tags::text ILIKE %s)")
            where_params.append(tag)
            where_params.append(f"%{tag}%")
        if min_conf > 0:
            where_parts.append("(meta->>'confidence')::float >= %s")
            where_params.append(min_conf)

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # Build display SQL for the UI
        display_parts = []
        if source:    display_parts.append(f"source = '{source}'")
        if tag:       display_parts.append(f"'{tag}' = ANY(tags)")
        if min_conf > 0: display_parts.append(f"(meta->>'confidence')::float >= {min_conf:.2f}")
        where_display = ("WHERE " + " AND ".join(display_parts)) if display_parts else ""

        SQL = f"""-- Metadata pre-filter + vector search
SET {'hnsw.ef_search' if INDEX_TYPE == 'hnsw' else 'vchordrq.probes'} = '{probes}';

SELECT id, content, source, tags,
       (meta->>'confidence')::float AS confidence,
       embedding <=> $1::vector    AS dist
FROM   {tbl}
{where_display}
ORDER  BY embedding <=> $1::vector
LIMIT  {topk};"""

        t0 = time.perf_counter()
        # Full param list:
        #   [vs]           -> embedding <=> %s::vector AS dist  (SELECT)
        #   where_params   -> WHERE conditions
        #   [vs, topk]     -> ORDER BY embedding <=> %s, LIMIT %s
        all_params = [vs] + where_params + [vs, topk]
        cur.execute(f"""
            SELECT id, content, source, tags,
                   (meta->>'confidence')::float AS confidence,
                   embedding <=> %s::vector AS dist
            FROM   {tbl}
            {where_sql}
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s""", all_params)
        rows    = cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000

        cur.execute(f"SELECT COUNT(*) FROM {tbl} {where_sql}", where_params)
        total = cur.fetchone()[0]
        cur.close(); c.close()

        trace("metadata", source=source, tag=tag, min_conf=min_conf,
              total_matching=total, latency_ms=round(elapsed, 2))

        return jsonify({
            "filters":        {"source": source, "tag": tag, "min_conf": min_conf},
            "total_matching": total,
            "latency_ms":     round(elapsed, 2),
            "sql":            SQL,
            "results": [
                {"id": r[0], "content": r[1][:100], "source": r[2],
                 "tags": list(r[3]) if r[3] else [],
                 "confidence": round(float(r[4]), 2) if r[4] else None,
                 "dist": round(float(r[5]), 4)}
                for r in rows
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e), "detail": str(type(e).__name__)}), 500

@app.route("/api/concurrent")
def concurrent():
    global TABLE
    tbl = TABLE
    n_threads  = int(request.args.get("threads", 4))
    iters_each = int(request.args.get("iters", 20))
    probes     = int(request.args.get("probes", 10))
    SQL = f"""-- Concurrent load test: {n_threads} threads × {iters_each} queries each
-- Each thread runs independently with its own connection

SET vchordrq.probes = '{probes}';

SELECT id FROM {tbl}
ORDER BY embedding <=> $1::vector  -- unique random vector per query
LIMIT 10;

-- Python: threading.Thread × {n_threads}, psycopg2 connection per thread
-- Measures true concurrent QPS + per-request p50/p95/p99"""
    results = [None] * n_threads; errors = []

    def worker(idx):
        try:
            c = get_conn(); cur = c.cursor()
            set_search_param(cur, probes)
            lats = []
            for i in range(iters_each):
                q = rand_vec(idx * 1000 + i)
                t0 = time.perf_counter()
                cur.execute(f"SELECT id FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT 10", (vec_str(q),))
                cur.fetchall()
                lats.append((time.perf_counter() - t0) * 1000)
            results[idx] = lats; cur.close(); c.close()
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t_start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total_elapsed = time.perf_counter() - t_start

    if errors:
        return jsonify({"error": errors[0]}), 500

    all_lats = sorted(l for r in results if r for l in r)
    mean = statistics.mean(all_lats) if all_lats else 0
    conc_result = {
        "threads": n_threads, "total_queries": len(all_lats),
        "elapsed_s": round(total_elapsed, 2),
        "qps": round(len(all_lats)/total_elapsed, 1),
        "p50": pct(all_lats,50), "p95": pct(all_lats,95), "p99": pct(all_lats,99),
        "sql": SQL,
    }
    trace("concurrent", threads=n_threads, total_queries=len(all_lats),
          elapsed_s=round(total_elapsed,2), qps=conc_result["qps"],
          p50=conc_result["p50"], p95=conc_result["p95"], p99=conc_result["p99"])
    return jsonify(conc_result)


@app.route("/api/indexes")
def indexes():
    global TABLE
    tbl = TABLE
    SQL = """SELECT i.relname          AS index_name,
       am.amname           AS index_type,
       pg_size_pretty(pg_relation_size(ix.indexrelid)) AS size,
       ix.indisvalid       AS valid,
       t.relname           AS table_name,
       pg_get_indexdef(ix.indexrelid) AS definition
FROM   pg_index ix
JOIN   pg_class i  ON i.oid = ix.indexrelid
JOIN   pg_class t  ON t.oid = ix.indrelid
JOIN   pg_am    am ON am.oid = i.relam
WHERE  am.amname IN ('vchordrq','hnsw','ivfflat')
ORDER  BY t.relname, i.relname;"""
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute(SQL)
        rows = cur.fetchall()
        cur.close(); c.close()
        return jsonify({"indexes": [
            {"name":r[0],"type":r[1],"size":r[2],"valid":r[3],"table":r[4],"def":r[5]}
            for r in rows], "sql": SQL})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def metrics():
    global TABLE
    tbl = TABLE
    probes = int(request.args.get("probes", 10))
    topk   = int(request.args.get("topk", 5))
    SQL_COSINE = f"""-- Cosine distance  (uses vchordrq index, probes={probes})
SET vchordrq.probes = '{probes}';

SELECT id,
       embedding <=> $1::vector AS cosine_distance
FROM   {tbl}
ORDER  BY embedding <=> $1::vector
LIMIT  {topk};
-- <=> operator: cosine distance, 0=identical, 1=orthogonal, 2=opposite"""
    SQL_L2 = f"""-- L2 (Euclidean) distance  (sequential scan — no L2 index built)
SELECT id,
       embedding <-> $1::vector AS l2_distance
FROM   {tbl}
ORDER  BY embedding <-> $1::vector
LIMIT  {topk};
-- <-> operator: sqrt(sum((a-b)^2))
-- To add an L2 index:
-- CREATE INDEX ON {tbl}
--   USING vchordrq (embedding vector_l2_ops) ...;"""
    try:
        c = get_conn(); cur = c.cursor()
        q = rand_vec(7)
        result = {}

        set_search_param(cur, probes)
        t0 = time.perf_counter()
        cur.execute(f"SELECT id, embedding <=> %s::vector AS d FROM {tbl} ORDER BY embedding <=> %s::vector LIMIT %s",
                    (vec_str(q), vec_str(q), topk))
        rows = cur.fetchall()
        result["cosine"] = {"latency_ms": round((time.perf_counter()-t0)*1000,2), "operator": "<=>",
                             "top_distances": [round(float(r[1]),6) for r in rows], "sql": SQL_COSINE}

        t0 = time.perf_counter()
        cur.execute(f"SELECT id, embedding <-> %s::vector AS d FROM {tbl} ORDER BY embedding <-> %s::vector LIMIT %s",
                    (vec_str(q), vec_str(q), topk))
        rows = cur.fetchall()
        result["l2"] = {"latency_ms": round((time.perf_counter()-t0)*1000,2), "operator": "<->",
                        "top_distances": [round(float(r[1]),6) for r in rows],
                        "note": "sequential scan (no L2 index)", "sql": SQL_L2}

        cur.close(); c.close()
        for metric, r in result.items():
            trace("metrics", metric=metric, operator=r.get("operator"),
                  latency_ms=r.get("latency_ms"),
                  top1_dist=(r.get("top_distances") or [None])[0])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sdk")
def sdk():
    SQL_CONNECT = """-- Connect + query from Python (psycopg2 + pgvector)
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

conn = psycopg2.connect("postgresql://user:pass@host:5444/db")
register_vector(conn)
cur = conn.cursor()

# Set search parameters
cur.execute("SET vchordrq.probes = '10'")

# Build your query vector (or get from embedding model)
query_vec = np.array([...], dtype=np.float32)
vec_literal = "[" + ",".join(f"{x:.6g}" for x in query_vec) + "]"

# Search
cur.execute(
    "SELECT id, content FROM documents "
    "ORDER BY embedding <=> %s::vector LIMIT 10",
    (vec_literal,)
)
results = cur.fetchall()"""
    try:
        c = get_conn(); cur = c.cursor()
        cur.execute("SELECT version()")
        pg_ver = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','vchord') ORDER BY extname")
        exts = [{"name": r[0], "version": r[1]} for r in cur.fetchall()]
        cur.close(); c.close()
        return jsonify({
            "connected": True, "pg_version": pg_ver, "extensions": exts,
            "driver": "psycopg2", "pgvector_python": "pgvector.psycopg2",
            "sql": SQL_CONNECT,
            "frameworks": [
                {"name": "psycopg2 + pgvector", "status": "connected", "note": "Used by this demo"},
                {"name": "LangChain (langchain-postgres)", "status": "compatible", "note": "PGVector store"},
                {"name": "LlamaIndex (PGVectorStore)", "status": "compatible", "note": "Supports vchordrq via hnsw_kwargs"},
                {"name": "SQLAlchemy ORM", "status": "compatible", "note": "Vector column type via pgvector"},
                {"name": "asyncpg", "status": "compatible", "note": "Async high-performance"},
            ]
        })
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)}), 500


# ── Frontend ───────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VectorChord + EDB — Live Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#F4F3EF;--surface:#FFFFFF;--surface2:#EEEDEA;--border:#DDDBD5;--border2:#C8C5BD;
  --text:#1A1A18;--muted:#6B6860;--faint:#9B9890;
  --green:#1A7A4A;--green-bg:#EBF5EF;--green-bd:#B8DFC8;
  --blue:#1A5A9A;--blue-bg:#EBF0F9;
  --amber:#9A6A00;--amber-bg:#FDF6E3;
  --red:#9A2A1A;--red-bg:#FCECEA;
  --teal:#0F6E5A;--teal-bg:#E8F5F2;--teal-bd:#A8D8CE;
  --mono:'IBM Plex Mono',monospace;
  --sans:'IBM Plex Sans',sans-serif;
  --r:6px;--rl:10px;
}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.6}
.site-header{background:var(--surface);border-bottom:1px solid var(--border);padding:0 2rem;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.brand{font-weight:600;font-size:15px;display:flex;align-items:center;gap:10px}
.brand-vc{color:var(--teal)}.brand-edb{color:var(--blue)}
.status-dot{width:8px;height:8px;border-radius:50%;background:#ccc;flex-shrink:0}
.status-dot.ok{background:var(--green);box-shadow:0 0 6px rgba(26,122,74,.4)}
.status-dot.err{background:var(--red)}
.header-right{display:flex;align-items:center;gap:12px;font-size:12px;color:var(--muted)}
.ext-badge{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-family:var(--mono);font-size:11px}
.badge-vc{background:var(--teal-bg);border-color:var(--teal-bd);color:var(--teal)}
.badge-hnsw{background:var(--blue-bg);border-color:var(--blue);color:var(--blue)}
.table-switcher{display:flex;align-items:center;gap:8px}
.table-dropdown-wrap{position:relative}
.table-active-btn{display:flex;align-items:center;gap:6px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:5px 10px;font-family:var(--mono);font-size:12px;font-weight:500;color:var(--teal);cursor:pointer;white-space:nowrap;transition:all .15s}
.table-active-btn:hover{border-color:var(--teal);background:var(--teal-bg)}
.table-dropdown{position:absolute;top:calc(100% + 4px);right:0;background:var(--surface);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.1);z-index:999;min-width:280px;overflow:hidden}
.table-option{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;cursor:pointer;transition:background .1s;gap:12px}
.table-option:hover{background:var(--surface2)}
.table-option.active{background:var(--teal-bg)}
.table-option-left{display:flex;flex-direction:column;gap:2px}
.table-option-name{font-family:var(--mono);font-size:13px;font-weight:500;color:var(--text)}
.table-option.active .table-option-name{color:var(--teal)}
.table-option-meta{font-size:11px;color:var(--muted)}
.table-option-right{text-align:right}
.table-option-rows{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text)}
.table-option-size{font-size:11px;color:var(--faint)}
.table-switching{animation:pulse-bg .5s ease}
@keyframes pulse-bg{0%{background:var(--teal-bg)}100%{background:transparent}}
.page{max-width:1200px;margin:0 auto;padding:2rem}
.section{background:var(--surface);border:1px solid var(--border);border-radius:var(--rl);margin-bottom:1.5rem;overflow:hidden}
.section-header{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:12px;cursor:pointer;user-select:none}
.section-header:hover{background:var(--surface2)}
.section-num{font-family:var(--mono);font-size:11px;color:var(--faint);min-width:28px}
.section-title{font-weight:600;font-size:14px;flex:1}
.section-desc{font-size:12px;color:var(--muted);max-width:500px}
.section-body{padding:20px;display:none}
.section-body.open{display:block}
.chevron{color:var(--faint);font-size:12px;transition:transform .2s}
.chevron.open{transform:rotate(180deg)}
.ctrl-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.ctrl-label{font-size:12px;color:var(--muted);white-space:nowrap}
input[type=range]{width:120px;accent-color:var(--teal)}
input[type=range]+output{font-family:var(--mono);font-size:12px;color:var(--teal);min-width:36px}
.run-btn{background:var(--teal);color:#fff;border:none;padding:7px 18px;border-radius:var(--r);font-family:var(--sans);font-size:13px;font-weight:500;cursor:pointer;transition:opacity .15s}
.run-btn:hover{opacity:.85}
.run-btn.secondary{background:var(--surface);color:var(--text);border:1px solid var(--border)}
.run-btn.secondary:hover{background:var(--surface2)}
select,input[type=text]{background:var(--surface);border:1px solid var(--border);color:var(--text);font-family:var(--sans);font-size:13px;padding:6px 10px;border-radius:var(--r);outline:none}
select:focus,input[type=text]:focus{border-color:var(--teal)}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--surface2);border-radius:var(--r);padding:14px 16px}
.kpi-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}
.kpi-val{font-family:var(--mono);font-size:22px;font-weight:500;color:var(--text);line-height:1}
.kpi-val.green{color:var(--green)}.kpi-val.blue{color:var(--blue)}.kpi-val.amber{color:var(--amber)}
.kpi-sub{font-size:11px;color:var(--faint);margin-top:3px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{text-align:left;padding:8px 10px;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);border-bottom:1px solid var(--border)}
.tbl td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:middle}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:var(--surface2)}
code{font-family:var(--mono);font-size:12px;background:var(--surface2);padding:2px 5px;border-radius:3px;color:var(--teal)}
.result-list{display:flex;flex-direction:column}
.result-item{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid var(--border)}
.result-item:last-child{border-bottom:none}
.result-rank{font-family:var(--mono);font-size:11px;color:var(--faint);min-width:22px;padding-top:2px}
.result-score{font-family:var(--mono);font-size:11px;color:var(--teal);min-width:70px;text-align:right;padding-top:2px}
.result-body{flex:1}
.result-title{font-size:13px;color:var(--text);margin-bottom:3px}
.result-meta{display:flex;gap:6px;flex-wrap:wrap}
.badge{display:inline-block;font-size:11px;padding:2px 7px;border-radius:3px;font-weight:500}
.bg{background:var(--green-bg);color:var(--green)}.bb{background:var(--blue-bg);color:var(--blue)}
.ba{background:var(--amber-bg);color:var(--amber)}.bt{background:var(--teal-bg);color:var(--teal)}
.br{background:var(--red-bg);color:var(--red)}
.loading{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);padding:16px 0}
.spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--teal);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-box{background:var(--red-bg);border:1px solid #f0c0bb;border-radius:var(--r);padding:12px 16px;font-size:13px;color:var(--red)}
/* SQL PANEL */
.sql-panel{border:1px solid var(--border);border-radius:var(--r);overflow:hidden;margin-top:14px}
.sql-panel-header{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;background:var(--surface2);border-bottom:1px solid var(--border);cursor:pointer;user-select:none}
.sql-panel-header:hover{background:#E5E4DF}
.sql-label{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--teal);display:flex;align-items:center;gap:6px}
.sql-label::before{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--teal)}
.sql-toggle-hint{font-size:11px;color:var(--faint)}
.sql-body{display:none;background:#1B1F27;padding:0}
.sql-body.open{display:block}
.sql-pre{margin:0;padding:14px 16px;font-family:var(--mono);font-size:12px;line-height:1.8;color:#ABB2BF;overflow-x:auto;white-space:pre}
.sql-footer{display:flex;align-items:center;justify-content:flex-end;padding:7px 14px;background:#161A22;border-top:1px solid #2A2F3A}
.copy-btn{background:#2A2F3A;border:1px solid #3A4050;color:#8A95A8;font-family:var(--mono);font-size:11px;padding:4px 12px;border-radius:4px;cursor:pointer;transition:all .15s}
.copy-btn:hover{background:#3A4050;color:#ABB2BF}
.copy-btn.copied{color:#4EC9A0;border-color:#4EC9A0}
/* SQL syntax colors */
.s-kw{color:#C678DD}.s-fn{color:#61AFEF}.s-str{color:#98C379}.s-cm{color:#5C6370;font-style:italic}
.s-op{color:#56B6C2}.s-num{color:#D19A66}
/* chips */
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{background:var(--surface2);border:1px solid var(--border);color:var(--muted);font-size:12px;padding:4px 12px;border-radius:20px;cursor:pointer;transition:all .15s}
.chip.active{background:var(--teal-bg);border-color:var(--teal-bd);color:var(--teal)}
.chip:hover:not(.active){border-color:var(--border2);color:var(--text)}
.mbar-wrap{display:flex;flex-direction:column;gap:10px;margin-top:12px}
.mbar-row{display:grid;grid-template-columns:70px 1fr 80px;gap:10px;align-items:center;font-size:12px}
.mbar-label{color:var(--muted);font-family:var(--mono)}
.mbar{height:7px;border-radius:4px;background:var(--surface2)}
.mbar-fill{height:100%;border-radius:4px;transition:width .5s ease}
.mbar-val{font-family:var(--mono);font-size:12px;text-align:right}
.thread-vis{display:flex;gap:4px;flex-wrap:wrap;margin:10px 0}
.thread-box{width:34px;height:34px;border-radius:4px;background:var(--surface2);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;color:var(--faint);transition:all .25s}
.thread-box.active{background:var(--teal-bg);border-color:var(--teal-bd);color:var(--teal)}
.thread-box.done{background:var(--green-bg);border-color:var(--green-bd);color:var(--green)}
.query-log{margin-bottom:4px}
.query-log-header{display:flex;align-items:center;justify-content:space-between;padding:7px 12px;background:var(--surface2);border:1px solid var(--border);border-radius:var(--r) var(--r) 0 0;border-bottom:none}
.query-log-label{font-family:var(--mono);font-size:11px;color:var(--teal)}
.query-log-timer{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--amber)}

@media(max-width:700px){.kpi-row{grid-template-columns:1fr 1fr}.ctrl-row{flex-direction:column;align-items:flex-start}}

/* SDK tabs */
.sdk-tabs{display:flex;gap:4px;margin-bottom:14px;flex-wrap:wrap}
.sdk-tab{background:var(--surface2);border:1px solid var(--border);color:var(--muted);font-size:12px;padding:6px 14px;border-radius:var(--r);cursor:pointer;transition:all .15s;font-family:var(--mono)}
.sdk-tab.active{background:var(--teal-bg);border-color:var(--teal-bd);color:var(--teal)}
.sdk-tab:hover:not(.active){color:var(--text);border-color:var(--border2)}
.sdk-panel{display:none}.sdk-panel.active{display:block}
</style>
</head>
<body>

<header class="site-header">
  <div class="brand">
    <span class="brand-vc">VectorChord</span>
    <span style="color:var(--border)">×</span>
    <span class="brand-edb">EDB</span>
    <span style="color:var(--border);font-weight:300">|</span>
    <span style="font-weight:400;color:var(--muted);font-size:13px">Live Capability Demo</span>
  </div>
  <div class="header-right">
    <div class="table-switcher" id="table-switcher">
      <span style="font-size:11px;color:var(--muted)">table</span>
      <div class="table-dropdown-wrap">
        <button class="table-active-btn" id="table-active-btn" onclick="toggleTableDropdown()">
          <span id="table-active-name">scale_bench_docs</span>
          <span id="table-active-rows" style="color:var(--muted);font-size:11px"></span>
          <span style="color:var(--faint);font-size:10px">▾</span>
        </button>
        <div class="table-dropdown" id="table-dropdown" style="display:none">
          <div id="table-dropdown-list"></div>
        </div>
      </div>
    </div>
    <div style="width:1px;height:20px;background:var(--border)"></div>
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text" style="color:var(--muted)">connecting…</span>
    <span class="ext-badge badge-vc" id="index-type-badge">vchordrq</span>
    <span class="ext-badge" id="vchord-ver"></span>
    <span class="ext-badge" id="vector-ver"></span>
  </div>
</header>

<div class="page">

  <!-- 1. LATENCY -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">01</span>
      <span class="section-title">Retrieval Latency</span>
      <span class="section-desc">Real p50/p95/p99 from live queries — adjust probes and top-K</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row">
        <span class="ctrl-label" id="probes-param-label">vchordrq.probes</span>
        <input type="range" min="1" max="100" value="10" id="lat-probes" oninput="this.nextElementSibling.value=this.value">
        <output>10</output>
        <span class="ctrl-label" style="margin-left:8px">top-K</span>
        <input type="range" min="1" max="50" value="10" id="lat-k" oninput="this.nextElementSibling.value=this.value">
        <output>10</output>
        <span class="ctrl-label" style="margin-left:8px">iterations</span>
        <select id="lat-iters">
          <option value="20">20 (fast)</option>
          <option value="50" selected>50</option>
          <option value="100">100</option>
        </select>
        <button class="run-btn" onclick="runLatency()">▶ Run</button>
        <button class="run-btn secondary" onclick="runRecall()" id="recall-btn" style="margin-left:4px">▲ Recall@K</button>
      </div>
      <div id="recall-result" style="margin-bottom:10px"></div>
      <div id="lat-result"></div>
    </div>
  </div>

  <!-- 2. INDEX INFO -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">02</span>
      <span class="section-title">Index Info</span>
      <span class="section-desc">Active indexes on this EDB instance — type, size, validity</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <button class="run-btn secondary" onclick="loadIndexInfo()">↻ Refresh</button>
      <div id="index-result" style="margin-top:14px"></div>
    </div>
  </div>

  <!-- 3. HYBRID SEARCH -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">03</span>
      <span class="section-title">Hybrid Search</span>
      <span class="section-desc">Vector similarity + full-text fused with Reciprocal Rank Fusion</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row">
        <input type="text" id="hybrid-q" value="machine learning training" style="width:280px">
        <div class="chips" id="mode-chips">
          <div class="chip active" data-mode="hybrid" onclick="setMode(this)">Hybrid (RRF)</div>
          <div class="chip" data-mode="vector" onclick="setMode(this)">Vector only</div>
          <div class="chip" data-mode="keyword" onclick="setMode(this)">Keyword only</div>
        </div>
        <button class="run-btn" onclick="runHybrid()">▶ Search</button>
      </div>
      <div id="hybrid-result"></div>
    </div>
  </div>

  <!-- 4. METADATA -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">04</span>
      <span class="section-title">Metadata Filtering</span>
      <span class="section-desc">Pre-filter by typed columns and JSONB before vector search</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row" style="margin-bottom:10px">
        <span class="ctrl-label">Source</span>
        <div class="chips" id="src-chips">
          <div class="chip active" data-val="" onclick="setSrc(this)">All</div>
          <div class="chip" data-val="internal" onclick="setSrc(this)">internal</div>
          <div class="chip" data-val="external" onclick="setSrc(this)">external</div>
        </div>
      </div>
      <div class="ctrl-row" style="margin-bottom:10px">
        <span class="ctrl-label">Tag</span>
        <div class="chips" id="tag-chips">
          <div class="chip active" data-val="" onclick="setTag(this)">All</div>
          <div class="chip" data-val="ml" onclick="setTag(this)">ml</div>
          <div class="chip" data-val="training" onclick="setTag(this)">training</div>
          <div class="chip" data-val="infra" onclick="setTag(this)">infra</div>
          <div class="chip" data-val="security" onclick="setTag(this)">security</div>
          <div class="chip" data-val="research" onclick="setTag(this)">research</div>
          <div class="chip" data-val="python" onclick="setTag(this)">python</div>
        </div>
      </div>
      <div class="ctrl-row">
        <span class="ctrl-label">Min confidence</span>
        <input type="range" min="0" max="100" value="0" id="conf-slider" oninput="document.getElementById('conf-out').textContent=this.value+'%'">
        <span id="conf-out" style="font-family:var(--mono);font-size:12px;min-width:36px">0%</span>
        <button class="run-btn" onclick="runMetadata()">▶ Filter + Search</button>
      </div>
      <div id="meta-result"></div>
    </div>
  </div>

  <!-- 5. CONCURRENT -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">05</span>
      <span class="section-title">Scalability &mdash; Concurrent Load</span>
      <span class="section-desc">Real parallel threads, real DB connections, real QPS measurement</span>
      <span class="chevron open">&#9660;</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row">
        <span class="ctrl-label">Concurrent threads</span>
        <input type="range" min="1" max="16" value="4" id="conc-threads"
               oninput="document.getElementById('conc-threads-out').value=this.value">
        <output id="conc-threads-out">4</output>
        <span class="ctrl-label" style="margin-left:8px">Queries each</span>
        <select id="conc-iters">
          <option value="10">10</option>
          <option value="20" selected>20</option>
          <option value="50">50</option>
        </select>
        <button class="run-btn" id="conc-run-btn" onclick="runConcurrent()">&#9654; Run</button>
      </div>
      <div id="conc-result"></div>
    </div>
  </div>

  <!-- 6. INDEXES -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">06</span>
      <span class="section-title">Multiple Indexes</span>
      <span class="section-desc">All vector indexes active in this database</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <button class="run-btn secondary" onclick="loadIndexes()">↻ Refresh</button>
      <div id="indexes-result" style="margin-top:14px"></div>
    </div>
  </div>

  <!-- 7. METRICS -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">07</span>
      <span class="section-title">Distance Metrics</span>
      <span class="section-desc">Cosine vs L2 — actual distances from real vectors in your table</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row">
        <span class="ctrl-label">vchordrq.probes</span>
        <input type="range" min="1" max="50" value="10" id="met-probes" oninput="this.nextElementSibling.value=this.value">
        <output>10</output>
        <button class="run-btn" onclick="runMetrics()">▶ Query both metrics</button>
      </div>
      <div id="metrics-result"></div>
    </div>
  </div>

  <!-- 8. SDK -->
  <div class="section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num">08</span>
      <span class="section-title">SDK &amp; Setup Guide</span>
      <span class="section-desc">How to load data, build indexes, and connect from Python for any table</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open">
      <div id="sdk-result"></div>

      <!-- Setup guide tabs -->
      <div style="margin-top:20px;border-top:1px solid var(--border);padding-top:18px">
        <div style="font-weight:600;font-size:14px;margin-bottom:14px">📋 How to set up a new table for this demo</div>
        <div class="sdk-tabs" id="sdk-tabs">
          <button class="sdk-tab active" data-tab="load">1 · Load data</button>
          <button class="sdk-tab" data-tab="index">2 · Build index</button>
          <button class="sdk-tab" data-tab="query">3 · Query</button>
          <button class="sdk-tab" data-tab="frameworks">4 · Frameworks</button>
        </div>

        <div class="sdk-panel active" id="sdk-tab-load">
          <div style="font-size:13px;color:var(--muted);margin-bottom:10px">
            Use <code>data_loader.py</code> to load synthetic 1536-dim vectors into any scale tier.
            Each run is incremental — it resumes from the current row count.
          </div>
          <pre class="sql-pre" style="background:#1B1F27;border-radius:6px;padding:14px 16px;font-size:12px;line-height:1.85;color:#ABB2BF;overflow-x:auto"><span style="color:#5C6370"># Install dependencies</span>
pip install psycopg2-binary pgvector numpy

<span style="color:#5C6370"># Load 1M rows (~15 min)</span>
python data_loader.py \
  --dsn <span style="color:#98C379">"postgresql://enterprisedb:admin@localhost:5444/repsol"</span> \
  --target 1m

<span style="color:#5C6370"># Then add more rows to reach 5M (incremental)</span>
python data_loader.py --dsn <span style="color:#98C379">"..."</span> --target 5m

<span style="color:#5C6370"># And 10M</span>
python data_loader.py --dsn <span style="color:#98C379">"..."</span> --target 10m

<span style="color:#5C6370"># The loader uses PostgreSQL COPY for bulk inserts
# All rows go into the same table: scale_bench_docs
# The table switcher in the header shows all vector tables</span></pre>
        </div>

        <div class="sdk-panel" id="sdk-tab-index">
          <div style="font-size:13px;color:var(--muted);margin-bottom:10px">
            Build a <code>vchordrq</code> index after loading data. Adjust <code>lists</code> based on row count.
          </div>
          <pre class="sql-pre" style="background:#1B1F27;border-radius:6px;padding:14px 16px;font-size:12px;line-height:1.85;color:#ABB2BF;overflow-x:auto"><span style="color:#5C6370"># Auto-selects lists= based on row count</span>
python build_index.py \
  --dsn <span style="color:#98C379">"postgresql://enterprisedb:admin@localhost:5444/repsol"</span> \
  --threads 8

<span style="color:#5C6370">-- Or run SQL directly in psql:</span>

<span style="color:#5C6370">-- For 1M rows → lists=[2000]</span>
<span style="color:#C678DD">CREATE INDEX</span> scale_bench_hnsw <span style="color:#C678DD">ON</span> scale_bench_docs
  <span style="color:#C678DD">USING</span> <span style="color:#61AFEF">vchordrq</span> (embedding <span style="color:#61AFEF">vector_cosine_ops</span>)
  <span style="color:#C678DD">WITH</span> (options = <span style="color:#98C379">$$
residual_quantization = true
build.pin = 2
[build.internal]
lists = [2000]
spherical_centroids = true
build_threads = 8
$$</span>);

<span style="color:#5C6370">-- Recommended lists by scale:
-- &lt; 500K rows  → lists = []
--   1M rows    → lists = [2000]
--   5M rows    → lists = [4000]
--  10M rows    → lists = [8000]
--  50M rows    → lists = [40000]</span></pre>
        </div>

        <div class="sdk-panel" id="sdk-tab-query">
          <div style="font-size:13px;color:var(--muted);margin-bottom:10px">
            Query VectorChord from Python using <code>psycopg2</code> and <code>pgvector</code>.
          </div>
          <pre class="sql-pre" style="background:#1B1F27;border-radius:6px;padding:14px 16px;font-size:12px;line-height:1.85;color:#ABB2BF;overflow-x:auto"><span style="color:#C678DD">import</span> psycopg2, numpy <span style="color:#C678DD">as</span> np
<span style="color:#C678DD">from</span> pgvector.psycopg2 <span style="color:#C678DD">import</span> register_vector

conn = psycopg2.<span style="color:#61AFEF">connect</span>(<span style="color:#98C379">"postgresql://enterprisedb:admin@localhost:5444/repsol"</span>)
conn.autocommit = <span style="color:#C678DD">True</span>
<span style="color:#61AFEF">register_vector</span>(conn)
cur = conn.cursor()

<span style="color:#5C6370"># Set probes — higher = better recall, lower QPS</span>
cur.<span style="color:#61AFEF">execute</span>(<span style="color:#98C379">"SET vchordrq.probes = '10'"</span>)

<span style="color:#5C6370"># Generate a query vector (replace with your real embedding)</span>
query_vec = np.random.<span style="color:#61AFEF">standard_normal</span>(1536).astype(<span style="color:#98C379">'float32'</span>)
query_vec /= np.linalg.<span style="color:#61AFEF">norm</span>(query_vec)
vec_str = <span style="color:#98C379">"["</span> + <span style="color:#98C379">","</span>.<span style="color:#61AFEF">join</span>(<span style="color:#98C379">f"</span>{x:.6g}<span style="color:#98C379">"</span> <span style="color:#C678DD">for</span> x <span style="color:#C678DD">in</span> query_vec) + <span style="color:#98C379">"]"</span>

<span style="color:#5C6370"># Nearest-neighbour search (cosine)</span>
cur.<span style="color:#61AFEF">execute</span>(<span style="color:#98C379">&quot;&quot;&quot;
    SELECT id, content, embedding &lt;=&gt; %s::vector AS dist
    FROM   scale_bench_docs
    ORDER  BY embedding &lt;=&gt; %s::vector
    LIMIT  10
&quot;&quot;&quot;</span>, (vec_str, vec_str))

<span style="color:#C678DD">for</span> row <span style="color:#C678DD">in</span> cur.<span style="color:#61AFEF">fetchall</span>():
    print(<span style="color:#98C379">f"</span>{row[2]:.4f}<span style="color:#98C379"> → </span>{row[1][:60]}<span style="color:#98C379">"</span>)</pre>
        </div>

        <div class="sdk-panel" id="sdk-tab-frameworks">
          <div style="font-size:13px;color:var(--muted);margin-bottom:12px">
            VectorChord works with any PostgreSQL-compatible framework.
          </div>
          <table class="tbl">
            <thead><tr><th>Framework</th><th>Status</th><th>Key setting</th></tr></thead>
            <tbody>
              <tr><td>psycopg2 + pgvector</td><td><span class="badge bg">Connected</span></td><td><code>SET vchordrq.probes = '10'</code></td></tr>
              <tr><td>LangChain (langchain-postgres)</td><td><span class="badge bt">Compatible</span></td><td>Uses pgvector PGVector store — vchordrq index is auto-used</td></tr>
              <tr><td>LlamaIndex (PGVectorStore)</td><td><span class="badge bt">Compatible</span></td><td>Standard PostgreSQL connection string</td></tr>
              <tr><td>SQLAlchemy ORM</td><td><span class="badge bt">Compatible</span></td><td><code>text("SET vchordrq.probes = '10'")</code> before queries</td></tr>
              <tr><td>asyncpg</td><td><span class="badge bt">Compatible</span></td><td>Same DSN, async <code>await conn.execute("SET vchordrq.probes...")</code></td></tr>
              <tr><td>JDBC (Java)</td><td><span class="badge bt">Compatible</span></td><td>Standard PostgreSQL JDBC driver</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- TUNING PANEL -->
  <div class="section" id="tuning-section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num" style="color:var(--amber)">⚙</span>
      <span class="section-title">Index Tuning</span>
      <span class="section-desc">Live query parameters — works for both vchordrq and pgvector HNSW</span>
      <span class="chevron open">▼</span>
    </div>
    <div class="section-body open" id="tuning-body">
      <div id="tuning-content">
        <div class="loading"><div class="spinner"></div>Loading tuning parameters…</div>
      </div>
    </div>
  </div>

  <!-- EXPLAIN ANALYZE -->
  <div class="section" id="explain-section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num" style="color:var(--green)">&#9650;</span>
      <span class="section-title">Explain Analyze</span>
      <span class="section-desc">Real query plan &#8212; confirms index is used and shows cost/buffers</span>
      <span class="chevron open">&#9660;</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row">
        <span class="ctrl-label" id="explain-probes-label">vchordrq.probes</span>
        <input type="range" min="1" max="100" value="10" id="explain-probes"
               oninput="this.nextElementSibling.value=this.value">
        <output>10</output>
        <span class="ctrl-label" style="margin-left:8px">top-K</span>
        <input type="range" min="1" max="50" value="10" id="explain-topk"
               oninput="this.nextElementSibling.value=this.value">
        <output>10</output>
        <button class="run-btn" onclick="runExplain()">&#9654; Explain Analyze</button>
      </div>
      <div id="explain-result"></div>
    </div>
  </div>

  <!-- QUERY EDITOR -->
  <div class="section" id="query-section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num" style="color:var(--blue)">&#9654;&#9135;</span>
      <span class="section-title">Query Editor</span>
      <span class="section-desc">Run any SELECT against the active table &#8212; full vectors included</span>
      <span class="chevron open">&#9660;</span>
    </div>
    <div class="section-body open">

      <!-- Full query generator -->
      <div style="margin-bottom:16px;padding:14px 16px;background:var(--surface2);border-radius:var(--r);border:1px solid var(--border)">
        <div style="font-size:12px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px">Generate full executable query</div>
        <div class="ctrl-row" style="margin-bottom:10px">
          <span class="ctrl-label">Mode</span>
          <div class="chips" id="fq-mode-chips">
            <div class="chip active" data-mode="vector" onclick="setFqMode(this)">Vector search</div>
            <div class="chip" data-mode="hybrid" onclick="setFqMode(this)">Hybrid (RRF)</div>
            <div class="chip" data-mode="metadata" onclick="setFqMode(this)">Metadata filter</div>
          </div>
          <span class="ctrl-label" style="margin-left:8px">top-K</span>
          <input type="range" min="1" max="50" value="10" id="fq-topk" oninput="this.nextElementSibling.value=this.value">
          <o>10</o>
          <button class="run-btn secondary" onclick="loadFullQuery()">Generate &#8594;</button>
        </div>
        <div style="font-size:11px;color:var(--faint);margin-bottom:8px">
          Generates a real query with an actual 1536-dim vector substituted in &#8212; copy and paste directly into psql
        </div>
        <div id="fq-result"></div>
      </div>

      <!-- Free-form editor -->
      <div style="font-size:12px;font-weight:500;color:var(--muted);margin-bottom:8px">
        SQL editor &#8212; SELECT / SET / SHOW / EXPLAIN only
      </div>
      <textarea id="qe-sql" spellcheck="false" style="
        width:100%;height:180px;font-family:var(--mono);font-size:12px;
        background:#1B1F27;color:#ABB2BF;border:1px solid var(--border);
        border-radius:var(--r);padding:12px 14px;resize:vertical;
        line-height:1.7;outline:none;box-sizing:border-box"
        placeholder="-- Example:&#10;SET vchordrq.probes = '10';&#10;&#10;SELECT id, content, embedding &lt;=&gt; '[0.1,...]'::vector AS dist&#10;FROM scale_bench_docs_1m&#10;ORDER BY embedding &lt;=&gt; '[0.1,...]'::vector&#10;LIMIT 10;"></textarea>
      <div class="ctrl-row" style="margin-top:10px">
        <button class="run-btn" id="qe-run-btn" onclick="runQuery()">&#9654; Run</button>
        <button class="run-btn secondary" onclick="loadFullQuery('vector', true)">&#8617; Load vector query</button>
        <button class="run-btn secondary" onclick="document.getElementById('qe-sql').value=''">&#215; Clear</button>
        <span style="font-size:12px;color:var(--muted);margin-left:auto" id="qe-status"></span>
      </div>
      <div id="qe-result" style="margin-top:12px"></div>
    </div>
  </div>

  <!-- TRACE LOG -->
  <div class="section" id="trace-section">
    <div class="section-header" onclick="toggle(this)">
      <span class="section-num" style="color:var(--amber)">&#9675;</span>
      <span class="section-title">Run Trace</span>
      <span class="section-desc">Every benchmark recorded &#8212; download as JSON or CSV for charting</span>
      <span class="chevron open">&#9660;</span>
    </div>
    <div class="section-body open">
      <div class="ctrl-row" style="margin-bottom:12px">
        <span class="ctrl-label">Filter</span>
        <select id="trace-filter-type" onchange="renderTrace()">
          <option value="">All types</option>
          <option value="latency">latency</option>
          <option value="concurrent">concurrent</option>
          <option value="hybrid">hybrid</option>
          <option value="metadata">metadata</option>
          <option value="metrics">metrics</option>
        </select>
        <select id="trace-filter-table" onchange="renderTrace()">
          <option value="">All tables</option>
        </select>
        <button class="run-btn secondary" onclick="downloadTrace('json')">&#8595; JSON</button>
        <button class="run-btn secondary" onclick="downloadTrace('csv')">&#8595; CSV</button>
        <button class="run-btn secondary" onclick="clearTrace()" style="margin-left:auto;color:var(--red)">Clear</button>
      </div>
      <div id="trace-stats" style="font-size:12px;color:var(--muted);margin-bottom:10px"></div>
      <div id="trace-table-wrap" style="overflow-x:auto">
        <table class="tbl" id="trace-tbl">
          <thead><tr>
            <th>time</th><th>type</th><th>table</th><th>index</th>
            <th>p50 ms</th><th>p95 ms</th><th>p99 ms</th><th>QPS</th>
            <th>latency ms</th><th>threads</th><th>detail</th>
          </tr></thead>
          <tbody id="trace-body"></tbody>
        </table>
        <p id="trace-empty" style="color:var(--muted);font-size:13px;display:none">No runs yet. Run any benchmark above to start tracing.</p>
      </div>
    </div>
  </div>

</div>

<script>
// ── Helpers ──────────────────────────────────────────────────────────────────
function toggle(hdr){
  const body=hdr.nextElementSibling, chev=hdr.querySelector('.chevron');
  body.classList.toggle('open'); chev.classList.toggle('open');
}
function loading(id){
  document.getElementById(id).innerHTML='<div class="loading"><div class="spinner"></div>Running live query against EDB…</div>';
}
function err(id,msg){
  document.getElementById(id).innerHTML=`<div class="error-box">Error: ${msg}</div>`;
}
async function api(path){ const r=await fetch(path); return r.json(); }
function badge(t,c){ return `<span class="badge ${c}">${t}</span>`; }

// ── SQL panel ────────────────────────────────────────────────────────────────
function sqlPanel(sql, openByDefault=false){
  if(!sql) return '';
  const esc = sql.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const highlighted = esc
    .replace(/\b(SET|SELECT|FROM|WHERE|ORDER BY|LIMIT|WITH|AS|LEFT JOIN|USING|JOIN|AND|OR|NOT|ON|IN|OVER|BY|DISTINCT|COALESCE|ROW_NUMBER|CREATE INDEX|USING|WITH)\b/g,'<span class="s-kw">$1</span>')
    .replace(/\b(vchordrq|hnsw|ivfflat|to_tsvector|to_tsquery|ts_rank|pg_size_pretty|pg_relation_size|pg_get_indexdef)\b/g,'<span class="s-fn">$1</span>')
    .replace(/(--[^\n]*)/g,'<span class="s-cm">$1</span>')
    .replace(/(\d+\.?\d*)/g,'<span class="s-num">$1</span>')
    .replace(/(&lt;=&gt;|&lt;-&gt;|&lt;#&gt;|@@|=&gt;)/g,'<span class="s-op">$&</span>');
  const uid = 'sql-' + Math.random().toString(36).slice(2,8);
  const open = openByDefault ? 'open' : '';
  return `
    <div class="sql-panel">
      <div class="sql-panel-header" onclick="toggleSql('${uid}')">
        <span class="sql-label">SQL executed</span>
        <span class="sql-toggle-hint" id="${uid}-hint">${open ? 'click to hide' : 'click to show'}</span>
      </div>
      <div class="sql-body ${open}" id="${uid}">
        <pre class="sql-pre">${highlighted}</pre>
        <div class="sql-footer">
          <button class="copy-btn" id="${uid}-copybtn" onclick="copySql('${uid}', this)">copy</button>
        </div>
      </div>
    </div>`;
}

function toggleSql(uid){
  const body=document.getElementById(uid);
  const hint=document.getElementById(uid+'-hint');
  const open=body.classList.toggle('open');
  hint.textContent = open ? 'click to hide' : 'click to show';
}

function copySql(uid, btn){
  const pre=document.getElementById(uid).querySelector('.sql-pre');
  const text=pre.textContent;
  navigator.clipboard.writeText(text).then(()=>{
    btn.textContent='copied!'; btn.classList.add('copied');
    setTimeout(()=>{ btn.textContent='copy'; btn.classList.remove('copied'); },2000);
  });
}

// ── STATUS + TABLE SWITCHER ──────────────────────────────────────────────────
async function loadStatus(){
  const d=await api('/api/status');
  const dot=document.getElementById('status-dot');
  const txt=document.getElementById('status-text');
  if(d.ok){
    dot.className='status-dot ok'; txt.textContent='Connected';
    if(d.extensions.vchord) document.getElementById('vchord-ver').textContent=`vchord ${d.extensions.vchord}`;
    if(d.extensions.vector) document.getElementById('vector-ver').textContent=`pgvector ${d.extensions.vector}`;
    // Update index type badge from current table
    if(d.index_type){
      const ib=document.getElementById('index-type-badge');
      if(ib){ ib.textContent=d.index_type; ib.className='ext-badge '+(d.index_type==='hnsw'?'badge-hnsw':'badge-vc'); }
      const pl=document.getElementById('probes-param-label');
      if(pl) pl.textContent = d.index_type==='hnsw'?'hnsw.ef_search':'vchordrq.probes';
    }
  } else {
    dot.className='status-dot err'; txt.textContent='Error: '+d.error;
  }
  loadTables();
}

async function loadTables(){
  const d=await api('/api/tables');
  if(d.error || !d.tables) return;
  const list=document.getElementById('table-dropdown-list');
  list.innerHTML = d.tables.map(t=>`
    <div class="table-option${t.name===d.active?' active':''}" onclick="switchTable('${t.name}')">
      <div class="table-option-left">
        <span class="table-option-name">${t.name}</span>
        <span class="table-option-meta">${t.vec_col} · ${t.vec_type} · <span style="color:${t.index_type==='hnsw'?'var(--blue)':'var(--teal)'}">${t.index_type||'no index'}</span></span>
      </div>
      <div class="table-option-right">
        <div class="table-option-rows">${t.rows.toLocaleString()}</div>
        <div class="table-option-size">${t.size}</div>
      </div>
    </div>`).join('');

  // Update active button
  const active = d.tables.find(t=>t.name===d.active);
  if(active){
    document.getElementById('table-active-name').textContent = active.name;
    document.getElementById('table-active-rows').textContent = active.rows.toLocaleString()+' rows';
  }
}

function toggleTableDropdown(){
  const dd=document.getElementById('table-dropdown');
  const isOpen = dd.style.display !== 'none';
  dd.style.display = isOpen ? 'none' : 'block';
  if(!isOpen) loadTables(); // refresh list when opening
}

// Close dropdown when clicking outside
document.addEventListener('click', e=>{
  if(!e.target.closest('.table-dropdown-wrap')){
    const dd=document.getElementById('table-dropdown');
    if(dd) dd.style.display='none';
  }
});

async function switchTable(name){
  document.getElementById('table-dropdown').style.display='none';
  const btn=document.getElementById('table-active-btn');
  btn.style.opacity='0.5';
  document.getElementById('table-active-name').textContent=name;
  document.getElementById('table-active-rows').textContent='switching…';

  const r=await fetch('/api/set_table',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({table:name})
  });
  const d=await r.json();
  btn.style.opacity='1';

  if(d.error){
    document.getElementById('table-active-name').textContent='Error';
    alert('Could not switch table: '+d.error);
    return;
  }

  document.getElementById('table-active-name').textContent=d.table;
  document.getElementById('table-active-rows').textContent=d.rows.toLocaleString()+' rows';
  btn.classList.add('table-switching');
  setTimeout(()=>btn.classList.remove('table-switching'),500);

  // Update index type badge
  const itype = d.index_type || 'vchordrq';
  const ibadge = document.getElementById('index-type-badge');
  if(ibadge){
    ibadge.textContent = itype;
    ibadge.className = 'ext-badge ' + (itype==='hnsw'?'badge-hnsw':itype==='vchordrq'?'badge-vc':'');
  }
  // Update probes label to match index type
  const plabel = document.getElementById('probes-param-label');
  if(plabel) plabel.textContent = itype==='hnsw' ? 'hnsw.ef_search' : 'vchordrq.probes';

  // Refresh all panels that show table-specific data
  loadIndexInfo();
  loadIndexes();
  loadTuning();

  // Show a toast
  showToast(`Switched to <b>${d.table}</b> (${itype}) — ${d.rows.toLocaleString()} rows`);
}

function showToast(html){
  let t=document.getElementById('toast');
  if(!t){
    t=document.createElement('div'); t.id='toast';
    t.style.cssText='position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#1A1A18;color:#fff;padding:10px 20px;border-radius:8px;font-size:13px;z-index:9999;opacity:0;transition:opacity .2s;pointer-events:none;white-space:nowrap';
    document.body.appendChild(t);
  }
  t.innerHTML=html; t.style.opacity='1';
  clearTimeout(t._timer);
  t._timer=setTimeout(()=>t.style.opacity='0',3000);
}

// ── 1. LATENCY ───────────────────────────────────────────────────────────────
async function runLatency(){
  const probes=document.getElementById('lat-probes').value;
  const k=document.getElementById('lat-k').value;
  const iters=document.getElementById('lat-iters').value;
  const el=document.getElementById('lat-result');

  const previewSql=`SET vchordrq.probes = '${probes}';\n\nSELECT id\nFROM   {TABLE}\nORDER  BY embedding <=> $1::vector  -- random unit vector, dim=1536\nLIMIT  ${k};\n\n-- Repeated ${iters}× with unique random query vectors\n-- Measures p50 / p95 / p99 / QPS`;
  el.innerHTML=`
    <div class="query-log">
      <div class="query-log-header">
        <span class="query-log-label">&#9654; Running ${iters} queries on EDB&hellip;</span>
        <span id="lat-timer" class="query-log-timer">0ms</span>
      </div>
      ${sqlPanel(previewSql, true)}
    </div>`;

  let ms=0;
  const timer=setInterval(()=>{
    ms+=50;
    const t=document.getElementById('lat-timer');
    if(t) t.textContent=ms>=1000?(ms/1000).toFixed(1)+'s':ms+'ms';
  },50);

  const d=await api(`/api/latency?probes=${probes}&topk=${k}&iters=${iters}`);
  clearInterval(timer);

  if(d.error) return err('lat-result',d.error);
  loadTrace();
  // Store full_sql for query editor
  if(d.full_sql) window._lastFullSql = d.full_sql;
  el.innerHTML=`
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">p50</div><div class="kpi-val green">${d.p50}<span style="font-size:13px;font-weight:400;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">p95</div><div class="kpi-val">${d.p95}<span style="font-size:13px;font-weight:400;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">p99</div><div class="kpi-val amber">${d.p99}<span style="font-size:13px;font-weight:400;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">QPS</div><div class="kpi-val blue">${d.qps}</div><div class="kpi-sub">single client</div></div>
      <div class="kpi"><div class="kpi-label">probes</div><div class="kpi-val">${d.probes}</div></div>
      <div class="kpi"><div class="kpi-label">iterations</div><div class="kpi-val">${d.iters}</div></div>
    </div>
    <p style="font-size:12px;color:var(--muted);margin-bottom:4px">${d.iters} live queries &middot; completed in <b style="color:var(--teal)">${ms>=1000?(ms/1000).toFixed(1)+'s':ms+'ms'}</b> wall time</p>
    ${sqlPanel(d.sql_setup + '\n\n' + d.sql_query)}
    ${d.full_sql ? `<div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap"><span style="font-size:12px;color:var(--muted)">Full executable query (real vector):</span><button class="run-btn secondary" style="padding:4px 12px;font-size:11px" onclick="document.getElementById('qe-sql').value=window._lastFullSql||'';document.getElementById('query-section').scrollIntoView({behavior:'smooth'})">Send to editor &#8594;</button><button class="copy-btn" onclick="navigator.clipboard.writeText(window._lastFullSql||'')">copy</button></div>` : ''}`;
}


// ── 2. INDEX INFO ─────────────────────────────────────────────────────────────
async function loadIndexInfo(){
  loading('index-result');
  const d=await api('/api/index_info');
  if(d.error) return err('index-result',d.error);
  const rows=d.indexes.map(i=>`
    <tr>
      <td><code>${i.name}</code></td>
      <td>${badge(i.type, i.type==='vchordrq'?'bt':'bb')}</td>
      <td>${i.size}</td>
      <td>${i.valid ? badge('valid','bg') : badge('invalid','br')}</td>
    </tr>`).join('');
  document.getElementById('index-result').innerHTML=`
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">Rows</div><div class="kpi-val blue">${d.rows.toLocaleString()}</div></div>
      <div class="kpi"><div class="kpi-label">Table size</div><div class="kpi-val">${d.table_size}</div></div>
      <div class="kpi"><div class="kpi-label">Vector indexes</div><div class="kpi-val">${d.indexes.length}</div></div>
    </div>
    ${d.indexes.length ? `<table class="tbl"><thead><tr><th>Index</th><th>Type</th><th>Size</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>` : '<p style="color:var(--muted);font-size:13px">No vector indexes found. Run build_index.py first.</p>'}
    ${sqlPanel(d.sql)}`;
}

// ── 3. HYBRID ─────────────────────────────────────────────────────────────────
let hybridMode='hybrid';
function setMode(el){
  document.querySelectorAll('#mode-chips .chip').forEach(c=>c.classList.remove('active'));
  el.classList.add('active'); hybridMode=el.dataset.mode;
}
async function runHybrid(){
  const q=document.getElementById('hybrid-q').value;
  const el=document.getElementById('hybrid-result');

  // Show SQL + live timer immediately, before the fetch returns
  function buildSqlPreview(queryText, mode){
    const words=queryText.toLowerCase().split(/\s+/).filter(w=>w.length>2);
    const tsq=words.length?words.join(' & '):'data';
    if(mode==='vector') return 'SET vchordrq.probes = \'10\';\n\nSELECT id, content, source, tags,\n       embedding <=> $1::vector AS score\nFROM   {TABLE}\nORDER  BY embedding <=> $1::vector\nLIMIT  10;';
    if(mode==='keyword') return 'SELECT id, content, source, tags,\n       ts_rank(to_tsvector(\'english\', content),\n               to_tsquery(\'english\', \''+tsq+'\')) AS score\nFROM   {TABLE}\nWHERE  to_tsvector(\'english\', content)\n         @@ to_tsquery(\'english\', \''+tsq+'\')\nORDER  BY score DESC\nLIMIT  10;';
    return 'SET vchordrq.probes = \'10\';\n\nWITH vec AS (\n    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk\n    FROM   {TABLE} LIMIT 100\n),\nkw AS (\n    SELECT id, ROW_NUMBER() OVER (\n               ORDER BY ts_rank(to_tsvector(\'english\', content),\n                                to_tsquery(\'english\', \''+tsq+'\')) DESC\n           ) AS rnk\n    FROM   {TABLE}\n    WHERE  to_tsvector(\'english\', content)\n             @@ to_tsquery(\'english\', \''+tsq+'\')\n    LIMIT  100\n)\nSELECT d.id, d.content, d.source, d.tags,\n       (COALESCE(1.0/(60+v.rnk),0) + COALESCE(1.0/(60+k.rnk),0)) AS rrf_score\nFROM   {TABLE} d\nLEFT   JOIN vec v USING (id)\nLEFT   JOIN kw  k USING (id)\nWHERE  v.id IS NOT NULL OR k.id IS NOT NULL\nORDER  BY rrf_score DESC\nLIMIT  10;';
  }

  const preview=buildSqlPreview(q, hybridMode);
  el.innerHTML=`
    <div class="query-log">
      <div class="query-log-header">
        <span class="query-log-label">&#9654; Executing on EDB&hellip;</span>
        <span id="hybrid-timer" class="query-log-timer">0ms</span>
      </div>
      ${sqlPanel(preview, true)}
    </div>`;

  let ms=0;
  const timer=setInterval(()=>{
    ms+=50;
    const t=document.getElementById('hybrid-timer');
    if(t) t.textContent=ms>=1000?(ms/1000).toFixed(1)+'s':ms+'ms';
  },50);

  const d=await api(`/api/hybrid?q=${encodeURIComponent(q)}&mode=${hybridMode}`);
  clearInterval(timer);

  if(d.error) return err('hybrid-result',`${d.error} — after ${ms}ms`);
  const items=d.results.map((r,i)=>`
    <div class="result-item">
      <div class="result-rank">#${i+1}</div>
      <div class="result-body">
        <div class="result-title">${r.content}</div>
        <div class="result-meta">${badge(r.source,r.source==='internal'?'bg':'ba')}${(r.tags||[]).map(t=>badge(t,'bt')).join('')}</div>
      </div>
      <div class="result-score">${r.score.toFixed(4)}</div>
    </div>`).join('');
  el.innerHTML=`
    <div style="display:flex;gap:16px;margin-bottom:12px;font-size:12px;color:var(--muted)">
      <span>Mode: ${badge(d.mode,'bt')}</span>
      <span>Query time: <b style="color:var(--teal)">${d.latency_ms}ms</b></span>
      <span>${d.results.length} results</span>
    </div>
    ${d.results.length?`<div class="result-list">${items}</div>`:'<p style="color:var(--muted);font-size:13px">No keyword matches. Try "Chunk" or switch to vector-only mode.</p>'}
    ${sqlPanel(d.sql||preview)}`;
}


// ── 4. METADATA ───────────────────────────────────────────────────────────────
let metaSrc='', metaTag='';
function setSrc(el){ document.querySelectorAll('#src-chips .chip').forEach(c=>c.classList.remove('active')); el.classList.add('active'); metaSrc=el.dataset.val; }
function setTag(el){ document.querySelectorAll('#tag-chips .chip').forEach(c=>c.classList.remove('active')); el.classList.add('active'); metaTag=el.dataset.val; }
async function runMetadata(){
  const conf=document.getElementById('conf-slider').value/100;
  loading('meta-result');
  const d=await api(`/api/metadata?source=${metaSrc}&tag=${metaTag}&min_conf=${conf}`);
  if(d.error) return err('meta-result',d.error);
  const items=d.results.map((r,i)=>`
    <tr><td>#${i+1}</td>
    <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.content}</td>
    <td>${badge(r.source,r.source==='internal'?'bg':'ba')}</td>
    <td>${(r.tags||[]).map(t=>badge(t,'bt')).join(' ')}</td>
    <td><code>${r.confidence??'—'}</code></td>
    <td><code>${r.dist}</code></td></tr>`).join('');
  document.getElementById('meta-result').innerHTML=`
    <div style="display:flex;gap:16px;margin-bottom:12px;font-size:12px;color:var(--muted)">
      <span>Matching rows: <b>${d.total_matching.toLocaleString()}</b></span>
      <span>Query time: <b>${d.latency_ms}ms</b></span>
    </div>
    <table class="tbl"><thead><tr><th></th><th>Content</th><th>Source</th><th>Tags</th><th>Confidence</th><th>Distance</th></tr></thead><tbody>${items}</tbody></table>
    ${sqlPanel(d.sql)}`;
}

// ── 5. CONCURRENT ─────────────────────────────────────────────────────────────
async function runConcurrent(){
  const threads = +document.getElementById('conc-threads').value;
  const iters   = +document.getElementById('conc-iters').value;
  const probes  = document.getElementById('lat-probes')?.value || 10;
  const btn     = document.getElementById('conc-run-btn');
  btn.disabled  = true;
  btn.textContent = '⏳ Running…';
  loading('conc-result');

  const d = await api(`/api/concurrent?threads=${threads}&iters=${iters}&probes=${probes}`);
  btn.disabled    = false;
  btn.textContent = '▶ Run';

  if(d.error) return err('conc-result', d.error);

  document.getElementById('conc-result').innerHTML = `
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">QPS</div><div class="kpi-val blue">${d.qps}</div><div class="kpi-sub">${threads} threads</div></div>
      <div class="kpi"><div class="kpi-label">p50</div><div class="kpi-val green">${d.p50}<span style="font-size:13px;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">p95</div><div class="kpi-val">${d.p95}<span style="font-size:13px;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">p99</div><div class="kpi-val amber">${d.p99}<span style="font-size:13px;color:var(--muted)"> ms</span></div></div>
      <div class="kpi"><div class="kpi-label">Total queries</div><div class="kpi-val">${d.total_queries}</div></div>
      <div class="kpi"><div class="kpi-label">Wall time</div><div class="kpi-val">${d.elapsed_s}<span style="font-size:13px;color:var(--muted)"> s</span></div></div>
    </div>
    ${sqlPanel(d.sql)}`;
  loadTrace();
}


// ── 6. INDEXES ────────────────────────────────────────────────────────────────
async function loadIndexes(){
  loading('indexes-result');
  const d=await api('/api/indexes');
  if(d.error) return err('indexes-result',d.error);
  if(!d.indexes.length){
    document.getElementById('indexes-result').innerHTML='<p style="color:var(--muted);font-size:13px">No vchordrq/hnsw/ivfflat indexes found.</p>'+sqlPanel(d.sql);
    return;
  }
  const rows=d.indexes.map(i=>`
    <tr>
      <td><code>${i.name}</code></td>
      <td>${badge(i.type,i.type==='vchordrq'?'bt':i.type==='hnsw'?'bb':'ba')}</td>
      <td><code>${i.table}</code></td>
      <td>${i.size}</td>
      <td>${i.valid ? badge('valid','bg') : badge('invalid','br')}</td>
    </tr>`).join('');
  document.getElementById('indexes-result').innerHTML=`
    <table class="tbl"><thead><tr><th>Name</th><th>Type</th><th>Table</th><th>Size</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>
    ${sqlPanel(d.sql)}`;
}

// ── 7. METRICS ────────────────────────────────────────────────────────────────
async function runMetrics(){
  const probes=document.getElementById('met-probes').value;
  loading('metrics-result');
  const d=await api(`/api/metrics?probes=${probes}`);
  if(d.error) return err('metrics-result',d.error);
  const maxD=Math.max(...(d.cosine?.top_distances||[1]),...(d.l2?.top_distances||[1]));
  function metBlock(key,color,label){
    const m=d[key]; if(!m) return '';
    const bars=m.top_distances.map((dist,i)=>`
      <div class="mbar-row">
        <span class="mbar-label">rank ${i+1}</span>
        <div class="mbar"><div class="mbar-fill" style="width:${Math.min(100,(dist/maxD)*100).toFixed(1)}%;background:${color}"></div></div>
        <span class="mbar-val">${dist.toFixed(6)}</span>
      </div>`).join('');
    return `<div style="flex:1;min-width:260px">
      <div style="font-weight:600;margin-bottom:3px">${label} <code>${m.operator}</code></div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:10px">${m.latency_ms}ms${m.note?' · <em>'+m.note+'</em>':''}</div>
      <div class="mbar-wrap">${bars}</div>
    </div>`;
  }
  document.getElementById('metrics-result').innerHTML=`
    <div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:4px">
      ${metBlock('cosine','var(--teal)','Cosine distance')}
      ${metBlock('l2','var(--blue)','L2 distance')}
    </div>
    ${sqlPanel(d.cosine.sql)}
    ${sqlPanel(d.l2.sql)}`;
}

// ── 8. SDK ────────────────────────────────────────────────────────────────────
async function loadSDK(){
  const d = await api('/api/sdk');
  const el = document.getElementById('sdk-result');
  if(d.error){ el.innerHTML = `<div class="error-box">${d.error}</div>`; return; }
  const exts = (d.extensions||[]).map(e=>`
    <div class="kpi"><div class="kpi-label">${e.name}</div><div class="kpi-val" style="font-size:16px">${e.version}</div></div>`).join('');
  el.innerHTML = `
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">Connection</div><div class="kpi-val green" style="font-size:16px">${d.connected?'Live':'Error'}</div></div>
      <div class="kpi"><div class="kpi-label">Driver</div><div class="kpi-val" style="font-size:14px">${d.driver}</div></div>
      ${exts}
    </div>`;

  // Wire up SDK tabs
  document.querySelectorAll('.sdk-tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.sdk-tab').forEach(b=>b.classList.remove('active'));
      document.querySelectorAll('.sdk-panel').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('sdk-tab-'+btn.dataset.tab).classList.add('active');
    });
  });
}


// ── TUNING PANEL ──────────────────────────────────────────────────────────────
const TUNING_DEFS = {
  vchordrq: [
    {
      param: "vchordrq.probes",
      label: "vchordrq.probes",
      type:  "range", min: 1, max: 200, step: 1, default: 10,
      desc:  "How many partitions to scan. Higher = better recall, lower QPS. Match length of lists= in your index."
    },
    {
      param: "vchordrq.epsilon",
      label: "vchordrq.epsilon",
      type:  "range", min: 0, max: 4.0, step: 0.1, default: 1.9,
      desc:  "RaBitQ lower-bound conservativeness. Lower = faster but less accurate. Default 1.9."
    },
    {
      param: "vchordrq.max_scan_tuples",
      label: "vchordrq.max_scan_tuples",
      type:  "number", min: -1, max: 10000000, default: -1,
      desc:  "Max tuples scanned before LIMIT is applied. -1 = unlimited. Useful with aggressive WHERE filters."
    }
  ],
  hnsw: [
    {
      param: "hnsw.ef_search",
      label: "hnsw.ef_search",
      type:  "range", min: 10, max: 1000, step: 10, default: 100,
      desc:  "Dynamic candidate list size during search. Higher = better recall, lower QPS. Must be ≥ LIMIT."
    },
    {
      param: "hnsw.iterative_scan",
      label: "hnsw.iterative_scan",
      type:  "select", options: ["off", "relaxed_order", "strict_order"],
      desc:  "Enable iterative scan for filtered queries. 'relaxed_order' is recommended with WHERE clauses."
    }
  ]
};

const TUNING_INDEX_PARAMS = {
  vchordrq: [
    { key: "lists",                   desc: "Number of Voronoi partitions. 1M→[2000], 5M→[4000], 10M→[8000]" },
    { key: "residual_quantization",   desc: "Quantize residuals for better accuracy. Recommended true for cosine." },
    { key: "build_threads",           desc: "K-means parallelism. Set to CPU count." },
    { key: "spherical_centroids",     desc: "Use spherical K-means. Set true for cosine similarity models." },
    { key: "build.pin",               desc: "Cache hot index data in shared memory during build. 2=full cache." }
  ],
  hnsw: [
    { key: "m",                desc: "Graph connectivity. Higher = better recall + larger index. 16→1M, 32→10M+." },
    { key: "ef_construction",  desc: "Build-time candidate list. Higher = better index quality, slower build." }
  ]
};

async function loadTuning() {
  const d = await api('/api/tuning');
  if(d.error){ document.getElementById('tuning-content').innerHTML = `<div class="error-box">${d.error}</div>`; return; }

  const itype = d.index_type || 'vchordrq';
  const defs  = TUNING_DEFS[itype] || TUNING_DEFS.vchordrq;
  const iparams = TUNING_INDEX_PARAMS[itype] || [];

  // Parse index definition to extract build params
  const idxDef = d.params.index_def || '';
  let idxParamsHtml = '';
  if(idxDef) {
    const paramMatches = idxDef.match(/WITH \((.+)\)/i);
    const paramStr = paramMatches ? paramMatches[1] : '';
    idxParamsHtml = `
      <div style="margin-bottom:16px">
        <div style="font-size:12px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px">
          Index build parameters
          ${badge(itype, itype==='hnsw'?'bb':'bt')}
        </div>
        <div class="code-block" style="font-size:11px;line-height:1.7">${idxDef.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>
        <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px">
          ${iparams.map(ip => `<div style="display:flex;gap:10px;font-size:12px">
            <span style="font-family:var(--mono);color:var(--teal);min-width:160px">${ip.key}</span>
            <span style="color:var(--muted)">${ip.desc}</span>
          </div>`).join('')}
        </div>
      </div>`;
  }

  // Build query param controls
  const controls = defs.map(def => {
    const curVal = d.params[def.param] || def.default;
    const numVal = parseFloat(curVal) || def.default;
    let input = '';
    if(def.type === 'range') {
      input = `
        <div style="display:flex;align-items:center;gap:10px;flex:1">
          <input type="range" min="${def.min}" max="${def.max}" step="${def.step||1}"
                 value="${numVal}" id="tune-${def.param.replace(/\./g,'-')}"
                 style="flex:1"
                 oninput="document.getElementById('tune-out-${def.param.replace(/\./g,'-')}').textContent=this.value">
          <span id="tune-out-${def.param.replace(/\./g,'-')}" style="font-family:var(--mono);font-size:13px;min-width:44px;color:var(--teal)">${numVal}</span>
          <button class="run-btn secondary" style="padding:5px 12px;font-size:12px"
                  onclick="applyTuning('${def.param}', document.getElementById('tune-${def.param.replace(/\./g,'-')}').value)">
            Apply
          </button>
        </div>`;
    } else if(def.type === 'select') {
      input = `
        <div style="display:flex;align-items:center;gap:10px;flex:1">
          <select id="tune-${def.param.replace(/\./g,'-')}" style="font-size:13px">
            ${(def.options||[]).map(o=>`<option value="${o}"${o===curVal?' selected':''}>${o}</option>`).join('')}
          </select>
          <button class="run-btn secondary" style="padding:5px 12px;font-size:12px"
                  onclick="applyTuning('${def.param}', document.getElementById('tune-${def.param.replace(/\./g,'-')}').value)">
            Apply
          </button>
        </div>`;
    } else {
      input = `
        <div style="display:flex;align-items:center;gap:10px;flex:1">
          <input type="number" value="${curVal}" min="${def.min}" max="${def.max}"
                 id="tune-${def.param.replace(/\./g,'-')}"
                 style="width:120px;font-size:13px">
          <button class="run-btn secondary" style="padding:5px 12px;font-size:12px"
                  onclick="applyTuning('${def.param}', document.getElementById('tune-${def.param.replace(/\./g,'-')}').value)">
            Apply
          </button>
        </div>`;
    }
    return `
      <div style="padding:12px 0;border-bottom:1px solid var(--border)">
        <div style="display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap">
          <div style="min-width:200px">
            <div style="font-family:var(--mono);font-size:13px;font-weight:500;color:var(--text);margin-bottom:3px">${def.label}</div>
            <div style="font-size:11px;color:var(--muted)">${def.desc}</div>
          </div>
          ${input}
        </div>
        <div id="tune-status-${def.param.replace(/\./g,'-')}" style="font-size:12px;margin-top:6px;color:var(--green);display:none"></div>
      </div>`;
  }).join('');

  document.getElementById('tuning-content').innerHTML = `
    ${idxParamsHtml}
    <div style="font-size:12px;font-weight:500;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:0">
      Query parameters — active for all queries in this session
    </div>
    ${controls}
    <p style="font-size:11px;color:var(--faint);margin-top:12px">
      These SET commands apply only to this session. Changes take effect immediately for all subsequent queries in the demo.
    </p>`;
}

async function applyTuning(param, value) {
  const statusEl = document.getElementById('tune-status-' + param.replace(/\./g, '-'));
  if(statusEl){ statusEl.style.display='block'; statusEl.style.color='var(--muted)'; statusEl.textContent='Applying…'; }
  const r = await fetch('/api/tuning', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({param, value})
  });
  const d = await r.json();
  if(statusEl){
    if(d.ok){
      statusEl.style.color='var(--green)';
      statusEl.textContent = `✓  ${param} = ${d.value}  (applied to session)`;
    } else {
      statusEl.style.color='var(--red)';
      statusEl.textContent = `Error: ${d.error}`;
    }
  }
  // Also update the probes slider in section 01 if we just changed the main search param
  if(param === 'vchordrq.probes' || param === 'hnsw.ef_search') {
    const latSlider = document.getElementById('lat-probes');
    if(latSlider) { latSlider.value = value; latSlider.nextElementSibling.value = value; }
  }
}



// ── EXPLAIN ANALYZE ───────────────────────────────────────────────────────────
async function runExplain() {
  const probes = document.getElementById('explain-probes').value;
  const topk   = document.getElementById('explain-topk').value;
  const el     = document.getElementById('explain-result');
  el.innerHTML = '<div class="loading"><div class="spinner"></div>Running EXPLAIN ANALYZE on EDB…</div>';

  const d = await api(`/api/explain?probes=${probes}&topk=${topk}`);
  if(d.error) return err('explain-result', d.error);

  // Status badge
  const statusBadge = d.seq_scan
    ? `<span class="badge br" style="font-size:12px;padding:4px 12px">&#9888; Seq Scan — index NOT used</span>`
    : `<span class="badge bg" style="font-size:12px;padding:4px 12px">&#10003; Index used</span>`;

  // Highlight key lines in the plan
  function highlightPlan(text) {
    return text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      // Index scans — green
      .replace(/(Index Scan[^\n]*)/g, '<span style="color:#1D9E75;font-weight:500">$1</span>')
      // vchordrq / hnsw specific
      .replace(/(Custom Scan \(vchordrq\)[^\n]*)/g, '<span style="color:#1D9E75;font-weight:500">$1</span>')
      .replace(/(Custom Scan \(hnsw\)[^\n]*)/g, '<span style="color:#1D9E75;font-weight:500">$1</span>')
      // Seq scan — red warning
      .replace(/((?:Parallel )?Seq Scan[^\n]*)/g, '<span style="color:#E24B4A;font-weight:500">$1</span>')
      // Timing lines — highlight actual time
      .replace(/(actual time=[\d.]+\.\.[\d.]+)/g, '<span style="color:#BA7517">$1</span>')
      // Rows
      .replace(/(rows=\d+)/g, '<span style="color:#378ADD">$1</span>')
      // Buffers
      .replace(/(Buffers:[^\n]*)/g, '<span style="color:var(--muted)">$1</span>')
      // Planning / Execution time
      .replace(/(Planning Time:[^\n]*)/g,  '<span style="color:var(--muted)">$1</span>')
      .replace(/(Execution Time:[^\n]*)/g, '<span style="color:#1D9E75;font-weight:500">$1</span>');
  }

  // Extract execution time from plan
  const execMatch = d.plan.match(/Execution Time:\s*([\d.]+)\s*ms/);
  const planMatch = d.plan.match(/Planning Time:\s*([\d.]+)\s*ms/);
  const execMs    = execMatch ? execMatch[1] : '—';
  const planMs    = planMatch ? planMatch[1] : '—';

  el.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap">
      ${statusBadge}
      <span style="font-size:12px;color:var(--muted)">
        Table: <b>${d.table}</b> &nbsp;|&nbsp;
        Index: <b style="color:${d.index_type==='hnsw'?'var(--blue)':'var(--teal)'}">${d.index_type}</b> &nbsp;|&nbsp;
        Execution: <b>${execMs}ms</b> &nbsp;|&nbsp;
        Planning: <b>${planMs}ms</b>
      </span>
    </div>
    ${d.warning ? `<div class="error-box" style="margin-bottom:12px">${d.warning}</div>` : ''}
    <div style="font-size:12px;color:var(--muted);margin-bottom:6px;font-weight:500">
      EXPLAIN (ANALYZE, BUFFERS) — top-${d.topk} &nbsp;·&nbsp;
      ${d.index_type==='hnsw'?'hnsw.ef_search':'vchordrq.probes'}=${d.probes}
    </div>
    <pre class="sql-pre" style="background:#1B1F27;border-radius:6px;padding:14px 16px;font-size:11px;line-height:1.7;overflow-x:auto;white-space:pre">${highlightPlan(d.plan)}</pre>
    <div style="margin-top:10px">
      <div style="font-size:12px;color:var(--muted);margin-bottom:6px;font-weight:500">Cost estimate (no ANALYZE)</div>
      <pre class="sql-pre" style="background:#1B1F27;border-radius:6px;padding:10px 16px;font-size:11px;line-height:1.7;overflow-x:auto;white-space:pre">${highlightPlan(d.cost_plan)}</pre>
    </div>`;

  loadTrace();
}


// ── QUERY EDITOR ──────────────────────────────────────────────────────────────
let fqMode = 'vector';

function setFqMode(el) {
  document.querySelectorAll('#fq-mode-chips .chip').forEach(c=>c.classList.remove('active'));
  el.classList.add('active');
  fqMode = el.dataset.mode;
}

async function loadFullQuery(mode, intoEditor) {
  const m    = mode || fqMode;
  const topk = document.getElementById('fq-topk')?.value || 10;
  const probes = document.getElementById('lat-probes')?.value || 10;
  const d    = await api(`/api/full_query?mode=${m}&topk=${topk}&probes=${probes}`);
  if(d.error) {
    document.getElementById('fq-result').innerHTML = `<div class="error-box">${d.error}</div>`;
    return;
  }

  if(intoEditor) {
    document.getElementById('qe-sql').value = d.sql;
    document.getElementById('qe-sql').focus();
    return;
  }

  document.getElementById('fq-result').innerHTML = `
    <div style="font-size:11px;color:var(--faint);margin-bottom:6px">
      Table: <b>${d.table}</b> &nbsp;·&nbsp;
      Index: <b style="color:${d.index_type==='hnsw'?'var(--blue)':'var(--teal)'}">${d.index_type}</b> &nbsp;·&nbsp;
      dim=${d.vector_dim} &nbsp;·&nbsp; top-${d.topk}
    </div>
    <div style="position:relative">
      <pre class="sql-pre" id="fq-code" style="background:#1B1F27;border-radius:6px;padding:12px 14px;font-size:11px;line-height:1.7;overflow-x:auto;white-space:pre;max-height:220px;overflow-y:auto">${d.sql.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</pre>
      <div style="display:flex;gap:8px;margin-top:6px">
        <button class="copy-btn" onclick="copyFq(this)">copy</button>
        <button class="copy-btn" onclick="sendToEditor()">send to editor &#8594;</button>
      </div>
    </div>`;
  // Store for copy/editor
  window._lastFullSql = d.sql;
}

function copyFq(btn) {
  navigator.clipboard.writeText(window._lastFullSql||'').then(()=>{
    btn.textContent='copied!'; setTimeout(()=>btn.textContent='copy',2000);
  });
}

function sendToEditor() {
  if(window._lastFullSql) {
    document.getElementById('qe-sql').value = window._lastFullSql;
    document.getElementById('qe-sql').scrollIntoView({behavior:'smooth'});
    document.getElementById('qe-sql').focus();
  }
}

async function runQuery() {
  const sql = document.getElementById('qe-sql').value.trim();
  if(!sql) return;
  const btn    = document.getElementById('qe-run-btn');
  const status = document.getElementById('qe-status');
  const result = document.getElementById('qe-result');
  btn.disabled = true; btn.textContent = '⏳ Running…';
  status.textContent = '';
  result.innerHTML   = '<div class="loading"><div class="spinner"></div>Executing on EDB…</div>';

  const d = await fetch('/api/query', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({sql, timeout_ms: 15000})
  }).then(r=>r.json());

  btn.disabled = false; btn.textContent = '▶ Run';

  if(d.error) {
    result.innerHTML = `<div class="error-box">${d.error}</div>`;
    status.textContent = 'Error';
    return;
  }

  status.innerHTML = `<span style="color:var(--green)">${d.row_count} row${d.row_count!==1?'s':''} &nbsp;·&nbsp; ${d.latency_ms}ms</span>`;
  loadTrace();

  if(!d.columns.length) {
    result.innerHTML = `<p style="color:var(--muted);font-size:13px">Query executed successfully (no rows returned).</p>`;
    return;
  }

  // Render result table
  const truncNote = d.truncated ? `<div style="font-size:11px;color:var(--amber);margin-bottom:8px">Showing first 200 rows</div>` : '';
  const thead = `<tr>${d.columns.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const tbody = d.rows.map(row => `<tr>${row.map(v => {
    // Truncate long values (vectors) for display
    const disp = v.length > 60 ? v.slice(0,57)+'…' : v;
    return `<td style="font-family:var(--mono);font-size:11px;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${v.replace(/"/g,'&quot;')}">${disp}</td>`;
  }).join('')}</tr>`).join('');

  result.innerHTML = `
    ${truncNote}
    <div style="overflow-x:auto">
      <table class="tbl" style="font-size:12px">
        <thead>${thead}</thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>`;
}

// Show full SQL in the latency panel when results come back
const _origLatHtml = null;  // patch latency result to add full_sql panel


// ── RECALL ────────────────────────────────────────────────────────────────────
async function runRecall() {
  const probes  = document.getElementById('lat-probes').value;
  const topk    = document.getElementById('lat-k').value;
  const btn     = document.getElementById('recall-btn');
  const el      = document.getElementById('recall-result');
  btn.disabled  = true; btn.textContent = '⏳ Computing…';
  el.innerHTML  = '<div class="loading"><div class="spinner"></div>Running ANN vs exact brute-force on EDB… (may take ~30s)</div>';

  const d = await api(`/api/recall?probes=${probes}&topk=${topk}&samples=10`);
  btn.disabled  = false; btn.textContent = '▲ Recall@K';

  if(d.error) return err('recall-result', d.error);

  const pct     = d.recall_pct;
  const color   = pct >= 95 ? 'var(--green)' : pct >= 80 ? 'var(--amber)' : 'var(--red)';
  const label   = pct >= 95 ? 'excellent' : pct >= 80 ? 'good' : 'low — increase probes';

  // Mini bar chart of per-query recall
  const bars = d.per_query.map(v => {
    const w   = Math.round(v);
    const c   = v >= 95 ? '#1D9E75' : v >= 80 ? '#BA7517' : '#E24B4A';
    return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
      <div style="width:${w*2}px;height:8px;background:${c};border-radius:2px"></div>
      <span style="font-family:var(--mono);font-size:10px;color:var(--muted)">${v}%</span>
    </div>`;
  }).join('');

  el.innerHTML = `
    <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:12px 16px;display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">Recall@${d.topk}</div>
        <div style="font-size:32px;font-weight:500;color:${color};line-height:1">${pct}%</div>
        <div style="font-size:11px;color:var(--muted);margin-top:4px">${label}</div>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">ANN latency</div>
        <div style="font-size:18px;font-weight:500;color:var(--teal)">${d.ann_ms}ms</div>
        <div style="font-size:11px;color:var(--muted)">avg · ${d.samples} queries</div>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">Exact latency</div>
        <div style="font-size:18px;font-weight:500">${d.exact_ms}ms</div>
        <div style="font-size:11px;color:var(--muted)">brute-force baseline</div>
      </div>
      <div>
        <div style="font-size:11px;color:var(--muted);margin-bottom:6px">per-query recall</div>
        ${bars}
      </div>
      <div style="flex:1;min-width:200px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:4px">how to improve</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.6">
          ${pct < 95 ? `&#x2191; probes from ${d.probes} to ${Math.min(200, d.probes*2)} &mdash; trades recall vs QPS<br>` : ''}
          probes=${d.probes} &nbsp;|&nbsp; table=${d.table} &nbsp;|&nbsp; ${d.index_type}
        </div>
      </div>
    </div>`;
  loadTrace();
}

// ── TRACE ─────────────────────────────────────────────────────────────────────
let traceData = [];

async function loadTrace() {
  const type  = document.getElementById('trace-filter-type')?.value  || '';
  const table = document.getElementById('trace-filter-table')?.value || '';
  let url = '/api/trace';
  const params = [];
  if(type)  params.push('type='  + encodeURIComponent(type));
  if(table) params.push('table=' + encodeURIComponent(table));
  if(params.length) url += '?' + params.join('&');
  const d = await api(url);
  if(d.error) return;
  traceData = d.entries || [];
  renderTrace();
  // Update table filter options
  const tables = [...new Set(traceData.map(e=>e.table).filter(Boolean))];
  const sel = document.getElementById('trace-filter-table');
  if(sel) {
    const cur = sel.value;
    sel.innerHTML = '<option value="">All tables</option>' +
      tables.map(t=>`<option value="${t}"${t===cur?' selected':''}>${t}</option>`).join('');
  }
}

function renderTrace() {
  const typeFilter  = document.getElementById('trace-filter-type')?.value  || '';
  const tableFilter = document.getElementById('trace-filter-table')?.value || '';
  let rows = traceData.slice().reverse(); // newest first
  if(typeFilter)  rows = rows.filter(r=>r.run_type===typeFilter);
  if(tableFilter) rows = rows.filter(r=>r.table===tableFilter);

  const tbody  = document.getElementById('trace-body');
  const empty  = document.getElementById('trace-empty');
  const stats  = document.getElementById('trace-stats');

  if(!rows.length) {
    if(tbody) tbody.innerHTML = '';
    if(empty) empty.style.display = 'block';
    if(stats) stats.textContent  = '';
    return;
  }
  if(empty) empty.style.display = 'none';

  // Stats summary
  const latencyRuns = rows.filter(r=>r.run_type==='latency');
  const bestQps = latencyRuns.length ? Math.max(...latencyRuns.map(r=>r.qps||0)).toFixed(1) : '—';
  const bestP50 = latencyRuns.length ? Math.min(...latencyRuns.map(r=>r.p50||Infinity)).toFixed(2) : '—';
  if(stats) stats.innerHTML = `
    <span style="margin-right:16px">${rows.length} runs recorded</span>
    ${latencyRuns.length ? `<span style="margin-right:16px">best p50: <b>${bestP50}ms</b></span><span>best QPS: <b>${bestQps}</b></span>` : ''}`;

  if(tbody) tbody.innerHTML = rows.slice(0,200).map(e => {
    const t    = e.ts ? e.ts.slice(11,19) : '—';
    const iclr = e.index_type==='hnsw' ? 'color:var(--blue)' : 'color:var(--teal)';
    let detail = '';
    if(e.run_type==='latency')    detail = `probes=${e.probes} top-${e.topk} ×${e.iters}`;
    if(e.run_type==='concurrent') detail = `${e.threads}t ×${e.total_queries} total`;
    if(e.run_type==='hybrid')     detail = `${e.mode}: "${(e.query||'').slice(0,30)}"`;
    if(e.run_type==='metadata')   detail = `src=${e.source||'all'} tag=${e.tag||'all'}`;
    if(e.run_type==='metrics')    detail = `${e.metric} ${e.operator}`;
    return `<tr>
      <td style="font-family:var(--mono);font-size:11px;color:var(--faint)">${t}</td>
      <td>${badge(e.run_type,'bt')}</td>
      <td style="font-family:var(--mono);font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis">${e.table||'—'}</td>
      <td style="font-family:var(--mono);font-size:11px;${iclr}">${e.index_type||'—'}</td>
      <td style="font-family:var(--mono);font-size:12px;color:var(--green)">${e.p50!=null?e.p50:'—'}</td>
      <td style="font-family:var(--mono);font-size:12px">${e.p95!=null?e.p95:'—'}</td>
      <td style="font-family:var(--mono);font-size:12px;color:var(--amber)">${e.p99!=null?e.p99:'—'}</td>
      <td style="font-family:var(--mono);font-size:12px;color:var(--blue)">${e.qps!=null?e.qps:'—'}</td>
      <td style="font-family:var(--mono);font-size:12px">${e.latency_ms!=null?e.latency_ms+'ms':'—'}</td>
      <td style="font-family:var(--mono);font-size:12px">${e.threads||'—'}</td>
      <td style="font-size:11px;color:var(--muted)">${detail}</td>
    </tr>`;
  }).join('');
}

function downloadTrace(fmt) {
  const type  = document.getElementById('trace-filter-type')?.value  || '';
  const table = document.getElementById('trace-filter-table')?.value || '';
  let url = `/api/trace?fmt=${fmt}`;
  if(type)  url += '&type='  + encodeURIComponent(type);
  if(table) url += '&table=' + encodeURIComponent(table);
  window.location.href = url;
}

async function clearTrace() {
  if(!confirm('Clear all trace data?')) return;
  await fetch('/api/trace', {method:'DELETE'});
  traceData = [];
  renderTrace();
  showToast('Trace log cleared');
}

// ── INIT ──────────────────────────────────────────────────────────────────────

loadStatus();
loadIndexInfo();
loadIndexes();
loadSDK();
loadTuning();
loadTrace();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VectorChord + EDB Live Demo")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--dim",  type=int, default=1536)
    args = parser.parse_args()
    DSN = args.dsn
    DIM = args.dim
    print(f"\n  VectorChord + EDB Live Demo")
    print(f"  DSN  : {args.dsn}")
    print(f"  Open : http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

