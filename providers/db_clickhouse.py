import asyncio

import clickhouse_connect
import pandas as pd
import datetime as dt

from apputils.log import write_log
from config.appconfig import  AI_FACTOR, SEVERITY
#from providers.web import get_rates_cbr

client = clickhouse_connect.get_client(host='localhost', database='app_storage', compress=False,
                                       username='default', password='z111111')

def db_prepare_tables(table):
    if table == 'cbr':
        client.command("alter table app_cbr delete where 1=1")
    if table == 'app':
        client.command("alter table app_row delete where 1=1")


def db_get_frames_by_facetype(ft, mean_over) -> pd.DataFrame:
    qry_str = f"select date_reg, workers,region, ratekey,usd,eur from app_row where typeface={ft} order by date_reg,region"
    if mean_over == AI_FACTOR.AIF_KR:
        qry_str = qry_str.replace(',usd,eur', '')
    elif mean_over == AI_FACTOR.AIF_EUR:
        qry_str = qry_str.replace(',ratekey,usd', '')
    elif mean_over == AI_FACTOR.AIF_USD:
        qry_str = qry_str.replace('ratekey,', '')
        qry_str = qry_str.replace(',eur', '')
    raw_data: pd.DataFrame = client.query_df(qry_str)
    return raw_data


def db_ret_okato():
    okato_sql = "select * from regions order by region"
    okato_data = client.query_df(okato_sql)
    return okato_data


def insert_data(df: pd.DataFrame):
    settings = {'async_insert': 1}
    client.insert_df(table='app_row', df=df, column_names=['date_reg', 'workers', 'okved',
                                                                 'region', 'typeface', 'ratekey',
                                                                 'usd',
                                                                 'eur'],
                     column_type_names=['Date', 'Int32', 'String', 'Int32', 'Int32', 'Float32',
                                              'Float32', 'Float32'], settings=settings)

def get_minmax():
    return client.query(query="select min(date_reg),max(date_reg) from app_row")


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
        query = ("alter table app_row update ratekey={key_r:Float32} where "
                 "date_reg={date_reg:DateTime}")  # ,usd=={usd:Float32},eur={eur:Float32}
        client.command(query, parameters=parameters)


def fill_glossary(parser,mindate=dt.datetime.strptime('01.01.2010', '%d.%m.%Y'), maxdate=dt.datetime.today()):
    asyncio.run(write_log(message=f'Load data from CBR:{dt.datetime.now()}', severity=SEVERITY.INFO))
    kv_dframe = parser.get_rates_cbr(mindate=mindate, maxdate=maxdate)
    db_prepare_tables('cbr')
    asyncio.run(
        write_log(message=f'Glossary:Write data to ClickHouse started:{dt.datetime.now()}', severity=SEVERITY.INFO))
    client.insert_df(table='app_cbr', df=kv_dframe, column_names=['date_', 'keyrate'],  # , 'usd', 'eur'],
                     column_type_names=['Date', 'Float32'])  # , 'Float32', 'Float32'])
    asyncio.run(write_log(message=f'Glossary:Finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
    return kv_dframe
