import numpy as np
import pandas as pd

from apputils.utils import create_intervals
from providers.db_clickhouse import *
import pandasql as ps


def df_recode_workers(df: pd.DataFrame):
    df_o = df
    cmpr = df_o['sworkers'].le(df_o['sworkers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['estimated'] = cmpr
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame, dbprovider, msp_type, multiclass=False):
    okatos = dbprovider.db_ret_okato()
    df_o = pd.DataFrame()
    df_mm = dbprovider.db_get_workers_limits(msp_type)
    for item in okatos.itertuples():
        asyncio.run(write_log(message=f'OKATO:{item.region}', severity=SEVERITY.INFO))
        subset = df[df['region'] == item.region]
        df_1 = df_recode_workers(subset)
        df_o = pd.concat([df_o, df_1], axis=0, ignore_index=True)
    return df_o


def df_fill_sors_apps(typeface, dates_frame,sors_frame, app_frame):
    if typeface == MSP_CLASS.MSP_UL:
        work_frame = sors_frame[sors_frame["msp_total"] > 0]
        work_frame = work_frame[['date_rep', 'okato', 'msp_total']]
    else:
        work_frame = sors_frame[sors_frame["il_total"] > 0]
        work_frame = work_frame[['date_rep', 'okato', 'il_total']]
    min_date = dates_frame['min_date'].values[0]
    max_date = dates_frame['max_date'].values[0]
    work_frame = work_frame[(work_frame["date_rep"] >= min_date) & (work_frame["date_rep"] <= max_date)]
    for item in work_frame.itertuples():
        app_frame.loc[(app_frame['date_reg'] == item.date_rep) & (app_frame['region'] == item.okato), "credits_mass"] = \
            item[3]
    app_frame["typeface"] = typeface.value
    return app_frame



