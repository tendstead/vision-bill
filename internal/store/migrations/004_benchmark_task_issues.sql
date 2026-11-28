-- Why a benchmark task is waiting or being retried (provider unreachable, model
-- missing or changed, ...). Additive and nullable: the previous release ignores it.
ALTER TABLE benchmark_tasks ADD COLUMN last_error text;
