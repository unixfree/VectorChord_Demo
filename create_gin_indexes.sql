-- ══════════════════════════════════════════════════════════════════
-- GIN indexes for hybrid search — VectorChord + pgvector tables
-- Run: psql -U enterprisedb -p 5444 -d repsol -f create_gin_indexes.sql
-- ══════════════════════════════════════════════════════════════════

-- ── VectorChord tables ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS scale_bench_docs_gin_fts
    ON scale_bench_docs
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS scale_bench_docs_1m_gin_fts
    ON scale_bench_docs_1m
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS scale_bench_docs_5m_gin_fts
    ON scale_bench_docs_5m
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS scale_bench_docs_10m_gin_fts
    ON scale_bench_docs_10m
    USING GIN (to_tsvector('english', content));

-- ── pgvector tables ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS bench_pgv_1m_gin_fts
    ON bench_pgv_1m
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS bench_pgv_5m_gin_fts
    ON bench_pgv_5m
    USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS bench_pgv_10m_gin_fts
    ON bench_pgv_10m
    USING GIN (to_tsvector('english', content));

-- ── tags TEXT[] — speeds up tag filter in metadata section ─────────
CREATE INDEX IF NOT EXISTS scale_bench_docs_1m_gin_tags
    ON scale_bench_docs_1m USING GIN (tags);

CREATE INDEX IF NOT EXISTS bench_pgv_1m_gin_tags
    ON bench_pgv_1m USING GIN (tags);

-- ── Verify all created ─────────────────────────────────────────────
SELECT t.relname AS table, i.relname AS index,
       pg_size_pretty(pg_relation_size(ix.indexrelid)) AS size,
       ix.indisvalid AS valid
FROM   pg_index ix
JOIN   pg_class i ON i.oid = ix.indexrelid
JOIN   pg_class t ON t.oid = ix.indrelid
WHERE  i.relname LIKE '%gin%'
  AND  t.relname LIKE '%bench%'
ORDER  BY t.relname, i.relname;
