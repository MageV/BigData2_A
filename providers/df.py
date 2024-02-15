import asyncio

import pandas as pd

from apputils.log import write_log
from config.appconfig import *
from providers.db_clickhouse import db_ret_okato


def df_recode_workers(df: pd.DataFrame):
    df_o=df
    cmpr = df_o['workers'].le(df_o['workers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['workers_ai'] = cmpr.reset_index()['workers']
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame):
    okatos = db_ret_okato(df)
    df_o=pd.DataFrame()
    for item in okatos.itertuples():
        asyncio.run(write_log(message=f'OKATO:{item[1]}', severity=SEVERITY.INFO))
        subset=df[df['region']==item[1]]
        df_1 = df_recode_workers(subset)
        df_o = pd.concat([df_o,df_1], axis=0, ignore_index=True)
    return df_o
