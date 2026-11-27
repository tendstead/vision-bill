-- Baseline schema (revision 0010), generated from a reference database.
CREATE FUNCTION public.vision_bill_check_receipt_image_id() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.image_id IS NULL
       OR (TG_OP = 'UPDATE' AND NEW.image_id IS NOT DISTINCT FROM OLD.image_id) THEN
        RETURN NEW;
    END IF;

    -- Serialize concurrent receipt writes for the same image even when
    -- historical duplicates prevent creation of the partial unique index.
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.image_id::text, 0));

    IF EXISTS (
        SELECT 1 FROM receipts WHERE image_id = NEW.image_id AND id <> NEW.id
    ) THEN
        RAISE EXCEPTION 'image_id % is already linked to a receipt', NEW.image_id
            USING ERRCODE = 'unique_violation';
    END IF;
    RETURN NEW;
END
$$;

CREATE TABLE public.benchmark_results (
    run_id uuid NOT NULL,
    model_id text NOT NULL,
    receipt_id uuid NOT NULL,
    success boolean NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE public.benchmark_runs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    status character varying(30) DEFAULT 'queued'::character varying NOT NULL,
    model_ids jsonb NOT NULL,
    receipt_ids jsonb NOT NULL,
    dataset_fingerprint text NOT NULL,
    model_digests jsonb DEFAULT '{}'::jsonb NOT NULL,
    prompt_version text NOT NULL,
    scoring_version text NOT NULL,
    request_timeout_seconds integer DEFAULT 300 NOT NULL,
    council_policy character varying(20) DEFAULT 'all'::character varying NOT NULL,
    council_absolute_threshold numeric,
    council_relative_threshold numeric,
    apply_council_flags boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    experiment_manifest jsonb
);

CREATE TABLE public.benchmark_summaries (
    run_id uuid NOT NULL,
    model_id text NOT NULL,
    model_digest text,
    completed integer DEFAULT 0 NOT NULL,
    succeeded integer DEFAULT 0 NOT NULL,
    failed integer DEFAULT 0 NOT NULL,
    total_score double precision DEFAULT 0 NOT NULL,
    total_confidence double precision DEFAULT 0 NOT NULL,
    total_attempts integer DEFAULT 0 NOT NULL,
    total_latency_ms double precision DEFAULT 0 NOT NULL,
    council_candidates integer DEFAULT 0 NOT NULL,
    council_findings integer DEFAULT 0 NOT NULL
);

CREATE TABLE public.benchmark_tasks (
    run_id uuid NOT NULL,
    model_id text NOT NULL,
    receipt_id uuid NOT NULL,
    status character varying(30) DEFAULT 'queued'::character varying NOT NULL,
    attempts integer DEFAULT 0 NOT NULL,
    leased_until timestamp with time zone,
    retry_at timestamp with time zone,
    started_at timestamp with time zone
);

CREATE TABLE public.collections (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    name character varying(100) NOT NULL,
    color character varying(20),
    start_date date,
    end_date date,
    active boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.images (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    original_filename character varying(255),
    media_type character varying(100),
    size_bytes bigint,
    image_path text,
    status character varying(20) DEFAULT 'pending'::character varying NOT NULL,
    error text,
    receipt_id uuid,
    bypass_review boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    analyzed_at timestamp with time zone,
    processing_at timestamp with time zone,
    user_id uuid,
    thumbnail_path text,
    additional_images jsonb DEFAULT '[]'::jsonb NOT NULL,
    model_id text,
    CONSTRAINT images_status_check CHECK (((status)::text = ANY ((ARRAY['pending'::character varying, 'processing'::character varying, 'analyzed'::character varying, 'failed'::character varying, 'timed_out'::character varying, 'unreadable'::character varying])::text[])))
);

CREATE TABLE public.item_category_aliases (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    merchant_key text NOT NULL,
    description_key text NOT NULL,
    category text NOT NULL
);

CREATE TABLE public.line_items (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    receipt_id uuid NOT NULL,
    description character varying(255) NOT NULL,
    quantity numeric(10,4) NOT NULL,
    unit_price numeric(14,2) NOT NULL,
    total_price numeric(14,2) NOT NULL,
    tags text[] DEFAULT '{}'::text[] NOT NULL,
    "position" integer DEFAULT 0 NOT NULL,
    spending_category text DEFAULT 'unknown'::text NOT NULL,
    category_source text DEFAULT 'unknown'::text NOT NULL,
    original_description text NOT NULL,
    CONSTRAINT line_items_category_source_check CHECK ((category_source = ANY (ARRAY['llm'::text, 'history'::text, 'user'::text, 'unknown'::text])))
);

CREATE TABLE public.receipt_collections (
    receipt_id uuid NOT NULL,
    collection_id uuid NOT NULL
);

CREATE TABLE public.receipts (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    confidence smallint DEFAULT 0 NOT NULL,
    merchant_name character varying(255) NOT NULL,
    merchant_address text,
    receipt_number character varying(100),
    date date NOT NULL,
    "time" time without time zone,
    currency character varying(10) DEFAULT 'USD'::character varying NOT NULL,
    subtotal numeric(14,2) DEFAULT 0 NOT NULL,
    discount_total numeric(14,2) DEFAULT 0 NOT NULL,
    tax_total numeric(14,2) DEFAULT 0 NOT NULL,
    tip numeric(14,2),
    total numeric(14,2) DEFAULT 0 NOT NULL,
    payment_method character varying(50) DEFAULT 'unknown'::character varying NOT NULL,
    status character varying(20) DEFAULT 'unverified'::character varying NOT NULL,
    verified boolean DEFAULT false NOT NULL,
    category character varying(50) DEFAULT 'other'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    user_id uuid,
    image_id uuid,
    CONSTRAINT receipts_status_check CHECK (((status)::text = ANY ((ARRAY['unverified'::character varying, 'verified'::character varying])::text[])))
);

CREATE TABLE public.tags (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    name character varying(100) NOT NULL
);

CREATE TABLE public.taxes (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    receipt_id uuid NOT NULL,
    name character varying(100) NOT NULL,
    rate numeric(6,4),
    amount numeric(14,2) NOT NULL
);

CREATE TABLE public.users (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    username character varying(100) NOT NULL,
    hashed_password text NOT NULL,
    is_admin boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);

ALTER TABLE ONLY public.benchmark_results
    ADD CONSTRAINT benchmark_results_pkey PRIMARY KEY (run_id, model_id, receipt_id);

ALTER TABLE ONLY public.benchmark_runs
    ADD CONSTRAINT benchmark_runs_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.benchmark_summaries
    ADD CONSTRAINT benchmark_summaries_pkey PRIMARY KEY (run_id, model_id);

ALTER TABLE ONLY public.benchmark_tasks
    ADD CONSTRAINT benchmark_tasks_pkey PRIMARY KEY (run_id, model_id, receipt_id);

ALTER TABLE ONLY public.collections
    ADD CONSTRAINT collections_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.item_category_aliases
    ADD CONSTRAINT item_category_aliases_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.item_category_aliases
    ADD CONSTRAINT item_category_aliases_user_id_merchant_key_description_key_key UNIQUE (user_id, merchant_key, description_key);

ALTER TABLE ONLY public.line_items
    ADD CONSTRAINT line_items_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.receipt_collections
    ADD CONSTRAINT receipt_collections_pkey PRIMARY KEY (receipt_id, collection_id);

ALTER TABLE ONLY public.receipts
    ADD CONSTRAINT receipts_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.tags
    ADD CONSTRAINT tags_name_key UNIQUE (name);

ALTER TABLE ONLY public.tags
    ADD CONSTRAINT tags_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.taxes
    ADD CONSTRAINT taxes_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);

CREATE INDEX ix_images_status_created_at ON public.images USING btree (status, created_at);

CREATE INDEX ix_images_user_id ON public.images USING btree (user_id);

CREATE INDEX ix_line_items_receipt_id ON public.line_items USING btree (receipt_id);

CREATE INDEX ix_receipt_collections_collection_id ON public.receipt_collections USING btree (collection_id);

CREATE INDEX ix_receipts_user_id ON public.receipts USING btree (user_id);

CREATE INDEX ix_receipts_verified_category ON public.receipts USING btree (category, currency) WHERE (((status)::text = 'verified'::text) AND (verified = true));

CREATE INDEX ix_receipts_verified_date_currency ON public.receipts USING btree (date DESC, currency) WHERE (((status)::text = 'verified'::text) AND (verified = true));

CREATE INDEX ix_receipts_verified_merchant ON public.receipts USING btree (merchant_name, currency) WHERE (((status)::text = 'verified'::text) AND (verified = true));

CREATE INDEX ix_receipts_verified_payment ON public.receipts USING btree (payment_method, currency) WHERE (((status)::text = 'verified'::text) AND (verified = true));

CREATE UNIQUE INDEX ux_collections_one_active_per_user ON public.collections USING btree (user_id) WHERE (active = true);

CREATE UNIQUE INDEX ux_receipts_image_id ON public.receipts USING btree (image_id) WHERE (image_id IS NOT NULL);

CREATE TRIGGER receipts_image_id_unique BEFORE INSERT OR UPDATE OF image_id ON public.receipts FOR EACH ROW EXECUTE FUNCTION public.vision_bill_check_receipt_image_id();

ALTER TABLE ONLY public.benchmark_results
    ADD CONSTRAINT benchmark_results_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.benchmark_runs(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.benchmark_summaries
    ADD CONSTRAINT benchmark_summaries_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.benchmark_runs(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.benchmark_tasks
    ADD CONSTRAINT benchmark_tasks_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES public.receipts(id);

ALTER TABLE ONLY public.benchmark_tasks
    ADD CONSTRAINT benchmark_tasks_run_id_fkey FOREIGN KEY (run_id) REFERENCES public.benchmark_runs(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.collections
    ADD CONSTRAINT collections_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES public.receipts(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.item_category_aliases
    ADD CONSTRAINT item_category_aliases_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.line_items
    ADD CONSTRAINT line_items_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES public.receipts(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.receipt_collections
    ADD CONSTRAINT receipt_collections_collection_id_fkey FOREIGN KEY (collection_id) REFERENCES public.collections(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.receipt_collections
    ADD CONSTRAINT receipt_collections_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES public.receipts(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.receipts
    ADD CONSTRAINT receipts_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.receipts
    ADD CONSTRAINT receipts_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.taxes
    ADD CONSTRAINT taxes_receipt_id_fkey FOREIGN KEY (receipt_id) REFERENCES public.receipts(id) ON DELETE CASCADE;

INSERT INTO public.tags (name) VALUES
    ('alcohol'), ('beverage'), ('coffee'), ('deposit'), ('electronics'), ('food'),
    ('fresh'), ('gift'), ('household'), ('hygiene'), ('office'), ('other'),
    ('pet'), ('service'), ('subscription'), ('travel')
ON CONFLICT (name) DO NOTHING;
