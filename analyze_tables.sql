-- Run once to give the planner accurate statistics
-- Without this, reltuples=0 and the planner always picks seqscan

ANALYZE scale_bench_docs;
ANALYZE scale_bench_docs_1m;
ANALYZE bench_pgv_1m;

-- Verify
SELECT relname,
       reltuples::bigint AS planner_row_estimate,
       last_analyze
FROM   pg_class c
JOIN   pg_stat_user_tables s USING (relname)
WHERE  relname IN ('scale_bench_docs','scale_bench_docs_1m','bench_pgv_1m')
ORDER  BY relname;
