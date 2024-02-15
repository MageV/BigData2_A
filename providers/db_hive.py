import asyncio

import pandas as pd
import datetime as dt

from pyhive import hive
from pyhive.hive import Connection

from apputils.log import write_log
from config.appconfig import AI_FACTOR, SEVERITY
from providers.web import get_rates_cbr

client = hive.connect('localhost')


def db_prepare_tables(table):
    if table == 'cbr':
        client.cursor().execute("delete from app_cbr where 1=1")
    if table == 'app':
        client.cursor().execute("delete from app_row where 1=1")


def db_get_frames_by_facetype(ft, mean_over) -> pd.DataFrame:
    qry_str = f"select date_reg, workers,region, ratekey,usd,eur from app_row where typeface={ft} order by date_reg,region"
    if mean_over == AI_FACTOR.AIF_KR:
        qry_str = qry_str.replace(',usd,eur', '')
    elif mean_over == AI_FACTOR.AIF_EUR:
        qry_str = qry_str.replace(',ratekey,usd', '')
    elif mean_over == AI_FACTOR.AIF_USD:
        qry_str = qry_str.replace('ratekey,', '')
        qry_str = qry_str.replace(',eur', '')
    raw_data: pd.DataFrame = pd.read_sql(qry_str, client)
    return raw_data


def db_ret_okato():
    okato_sql = "select * from regions order by region"
    okato_data = pd.read_sql(okato_sql, client)
    return okato_data


def insert_data(df: pd.DataFrame):
    settings = {'async_insert': 1}
    for item in df.itertuples():
        sql=f"insert into app_row values ({item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]},{item[7]},{item[8]});"
        client.cursor().execute(sql)


def get_minmax():
    return pd.read_sql("select min(date_reg),max(date_reg) from app_row",client)


def update_rows_kv(kvframe: pd.DataFrame):
    for item in kvframe.itertuples():
        date_reg = item[1]
        key_r = item[2]
        #  usd = item[3]
        #  eur = item[4]
        parameters = {'key_r': key_r,
                      #         'usd': usd,
                      #         'eur': eur,
                      'date_reg': date_reg}
        query = (f"update app_row set ratekey={key_r} where "
                 f"date_reg={date_reg}")  # ,usd=={usd:Float32},eur={eur:Float32}
        client.cursor().execute(query)


def fill_glossary(mindate=dt.datetime.strptime('01.01.2010', '%d.%m.%Y'), maxdate=dt.datetime.today()):
    asyncio.run(write_log(message=f'Load data from CBR:{dt.datetime.now()}', severity=SEVERITY.INFO))
    kv_dframe = get_rates_cbr(mindate=mindate, maxdate=maxdate)
    db_prepare_tables('cbr')
    asyncio.run(
        write_log(message=f'Glossary:Write data to ClickHouse started:{dt.datetime.now()}', severity=SEVERITY.INFO))
    for item in kv_dframe.itertuples():
        sql = f"insert into app_row values ({item[1]},{item[2]});"
        client.cursor().execute(sql)
    asyncio.run(write_log(message=f'Glossary:Finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
    return kv_dframe
