-- app_storage.app_cbr definition

CREATE TABLE app_storage.app_cbr
(

    `date_` Date,

    `keyrate` Float32,

    `usd` Float32,

    `eur` Float32
)
ENGINE = MergeTree
ORDER BY date_
SETTINGS index_granularity = 8192;
-- app_storage.app_row definition

CREATE TABLE app_storage.app_row
(

    `date_reg` Date,

    `workers` Int32,

    `region` Int32,

    `typeface` Int32,

    `credits_mass` Float32
)
ENGINE = MergeTree
ORDER BY date_reg
SETTINGS index_granularity = 8192;

-- app_storage.okato definition

CREATE TABLE app_storage.okato
(

    `okato_code` Int64,

    `regname` String
)
ENGINE = MergeTree
ORDER BY okato_code
SETTINGS index_granularity = 8192;

-- app_storage.sors definition

CREATE TABLE app_storage.sors
(

    `region` String,

    `total` Float64,

    `msp_total` Float64,

    `il_total` Float64,

    `date_rep` Date,

    `okato` Int32
)
ENGINE = MergeTree
ORDER BY date_rep
SETTINGS index_granularity = 8192;

-- app_storage.debt definition

CREATE TABLE app_storage.debt
(

    `region` String,

    `total` Float64,

    `msp_total` Float64,

    `il_total` Float64,

    `date_rep` Date,

    `okato` Int32
)
ENGINE = MergeTree
ORDER BY date_rep
SETTINGS index_granularity = 8192;

-- app_storage.reduced_app_view source

CREATE VIEW app_storage.reduced_app_view
(

    `date_reg` Date,

    `region` Int32,

    `typeface` Int32,

    `sworkers` Int64
)
AS SELECT
    date_reg,

    region,

    typeface,

    sum(workers) AS sworkers
FROM app_storage.app_row
WHERE app_row.typeface = 1
GROUP BY
    date_reg,

    region,

    typeface
UNION ALL
SELECT
    date_reg,

    region,

    typeface,

    sum(workers) AS sworkers
FROM app_storage.app_row
WHERE app_row.typeface = 0
GROUP BY
    date_reg,

    region,

    typeface;

    -- app_storage.regions source

CREATE VIEW app_storage.regions
(

    `region` Int32
)
AS SELECT DISTINCT region
FROM app_storage.app_row AS ar;

-- app_storage.regions source

CREATE VIEW app_storage.regions
(

    `region` Int32
)
AS SELECT DISTINCT region
FROM app_storage.app_row AS ar;