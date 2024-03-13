-- app_storage.app_cbr definition

CREATE TABLE IF NOT EXISTS app_storage.app_cbr
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

CREATE TABLE IF NOT EXISTS app_storage.app_row
(

    `date_reg` Date,

    `workers` Int32,

    `okved` String,

    `region` Int32,

    `typeface` Int32,

    `ratekey` Float32,

    `credits_mass` Float32
)
ENGINE = MergeTree
ORDER BY date_reg
SETTINGS index_granularity = 8192;

-- app_storage.okato definition

CREATE TABLE IF NOT EXISTS app_storage.okato
(

    `okato_code` Int64,

    `regname` String
)
ENGINE = MergeTree
ORDER BY okato_code
SETTINGS index_granularity = 8192;

-- app_storage.sors definition

CREATE TABLE  IF NOT EXISTS app_storage.sors
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

-- app_storage.regions source

CREATE VIEW IF NOT EXISTS app_storage.regions
(

    `region` Int32
) AS
SELECT DISTINCT region
FROM app_storage.app_row AS ar;

-- app_storage.serv_app_rows source

CREATE VIEW app_storage.serv_app_rows
(

    `date_reg` Date,

    `typeface` Int32
) AS
SELECT DISTINCT
    date_reg AS date_reg,

    typeface
FROM app_storage.app_row AS ar
ORDER BY date_reg ASC;

-- app_storage.serv_app_rows_reduced source

CREATE VIEW app_storage.serv_app_rows_reduced
(

    `date_reg` Date,

    `region` Int32,

    `typeface` Int32,

    `workers` Int64
) AS
SELECT
    date_reg,

    region,

    typeface,

    sum(workers) AS workers
FROM app_storage.app_row AS ar
GROUP BY
    date_reg,

    region,

    typeface
ORDER BY
    region ASC,

    date_reg ASC;

-- app_storage.serv_sors_regs source

CREATE VIEW app_storage.serv_sors_regs
(

    `okato_reg` Int32
) AS
SELECT DISTINCT okato AS okato_reg
FROM app_storage.sors AS ar;
