import numpy as np
import pandas as pd

from apputils.utils import create_intervals
from providers.db_clickhouse import *
import pandasql as ps


def df_recode_workers(df: pd.DataFrame, mmvalues, multiclass):
    df_o = df
#    if multiclass:
#        a_min = mmvalues[0]
#        a_max = mmvalues[1]
#        intervals=[i for i in range(a_min,a_max,10)]
#        res = create_intervals(intervals)
#        for i in range(len(res)):
#            df_o.loc[(df_o['workers'] >= res[i][0]) & (df_o['workers'] <= res[i][1]), "workers_ai"] = i
#    else:
    cmpr = df_o['workers'].le(df_o['workers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['workers_ai'] = cmpr.reset_index()['workers']
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame, dbprovider, msp_type, multiclass=False):
    okatos = dbprovider.db_ret_okato()
    df_o = pd.DataFrame()
    df_mm = dbprovider.db_get_workers_limits(msp_type)
    for item in okatos.itertuples():
        asyncio.run(write_log(message=f'OKATO:{item[1]}', severity=SEVERITY.INFO))
        subset = df[df['region'] == item[1]]
        df_1 = df_recode_workers(subset, df_mm, False)
        df_o = pd.concat([df_o, df_1], axis=0, ignore_index=True)
    return df_o


def df_fill_sors_apps(typeface, dates_frame, okatos_frame, sors_frame, app_frame):
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

