import asyncio
import gc

import pandas as pd
import pandasql as ps
import datetime as dt

from apputils.log import write_log
from config.appconfig import *


def db_prepare_tables(table):
    if table == 'cbr':
        click_client.command("alter table app_cbr delete where 1=1")
    if table == 'app':
        click_client.command("alter table app_row delete where 1=1")


def db_get_frames_by_facetype(ft, mean_over) -> pd.DataFrame:
    qry_str = f"select date_reg, workers,region, ratekey,usd,eur from app_row where typeface={ft} order by date_reg,region"
    if mean_over == AI_FACTOR.AIF_KR:
        qry_str = qry_str.replace(',usd,eur', '')
    elif mean_over == AI_FACTOR.AIF_EUR:
        qry_str = qry_str.replace(',ratekey,usd', '')
    elif mean_over == AI_FACTOR.AIF_USD:
        qry_str = qry_str.replace('ratekey,', '')
        qry_str = qry_str.replace(',eur', '')
    raw_data: pd.DataFrame = click_client.query_df(qry_str)
    return raw_data


def db_ret_okato():
    okato_sql = "select * from regions order by region"
    okato_data = click_client.query_df(okato_sql)
    return okato_data


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
