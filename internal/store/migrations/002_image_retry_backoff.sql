-- Timeouts are no longer final: a timed-out submission is retried with an
-- increasing delay (see store.MarkImageTimedOut). Both columns are additive
-- and defaulted, so the previous release keeps working on this schema.
ALTER TABLE images ADD COLUMN analysis_attempts integer DEFAULT 0 NOT NULL;
ALTER TABLE images ADD COLUMN retry_after timestamp with time zone;
