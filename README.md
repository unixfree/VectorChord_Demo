# VectorChord_Demo

0. 사전 준비
   - EPAS 18 설치 구성
   - pgvector, VectorChord Extension 설치 구성
     ```
     sudo dnf -y install edb-as18-vectorchord

     vi as18/data/postgresql.conf
     shared_preload_libraries = 'vchord, pg_stat_statements'

     sudo systemctl restart edb-as-18

     psql -d edb
     create extension vector;
     CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
     \dx
     ```
     
   - python 설치 및 필요 pkg 설치
     ```
     pip install numpy psycopg2-binary pgvector
     ```

1. data load
```
cd VectorChordDemo
python data_loader.py --dsn "postgresql://enterprisedb:enterprisedb@localhost:5444/edb" --target 1m --tablename scale_bench_docs_10m --workers 4
```

2. build index vchordrq
```
time python build_index.py --dsn "postgresql://enterprisedb:enterprisedb@localhost:5444/edb" --tablename scale_bench_docs_1m --threads 20 --mem 16GB
```
2-1.build index hnsw
```
time python build_index_pgvector.py --dsn "postgresql://enterprisedb:enterprisedb@localhost:5444/edb" --tablename scale_bench_docs_1m --m 16 --efc 200 --mem 10GB
```

3. search and index example ..
```
\timing
select count(*) from scale_bench_docs_1m;

SELECT id, content, source FROM scale_bench_docs_1m WHERE id != 1 ORDER BY embedding <=> (SELECT embedding FROM scale_bench_docs_1m WHERE id = 1) LIMIT 5;
\di+

SET max_parallel_maintenance_workers = 8;
SET max_parallel_workers = 8;
SET maintenance_work_mem = '10GB';

CREATE INDEX scale_bench_docs_1m_hnsw ON scale_bench_docs_1m USING hnsw ((embedding::vector(1536)) vector_cosine_ops) WITH (m = 16, ef_construction = 200);
\di+

SELECT id, content, source FROM scale_bench_docs_1m WHERE id != 1 ORDER BY embedding <=> (SELECT embedding FROM scale_bench_docs_1m WHERE id = 1) LIMIT 5;

DROP INDEX scale_bench_docs_1m_hnsw;
\di+

SET max_parallel_maintenance_workers = 8;
SET max_parallel_workers = 8;
SET maintenance_work_mem = '16GB';

CREATE INDEX scale_bench_docs_1m_vchordrq ON scale_bench_docs_1m 
USING vchordrq ((embedding::vector(1536)) vector_cosine_ops) WITH (options = $$
residual_quantization = true
[build.internal]
lists = [2000]
spherical_centroids = true
build_threads = 16
$$);
\di+

SET vchordrq.probes TO '10';
SELECT id, content, source FROM scale_bench_docs_1m WHERE id != 1 ORDER BY embedding <=> (SELECT embedding FROM scale_bench_docs_1m WHERE id = 1) LIMIT 5;

DROP INDEX scale_bench_docs_1m_vchordrq;
\di+
```
