import asyncio
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

import clickhouse_connect
import pandas as pd

from apputils.log import write_log
from config.appconfig import *


# from providers.web import get_rates_cbr

class DBConnector:

    def __init__(self):

        self.client = clickhouse_connect.get_client(host='localhost', database='app_storage', compress=False,
                                                    username='default', password='z111111')

    def db_prepare_tables(self, table):
        if table == PRE_TABLES.PT_CBR:
            self.client.command("alter table app_cbr delete where 1=1")
        if table == PRE_TABLES.PT_APP:
            self.client.command("alter table app_row delete where 1=1")
        if table == PRE_TABLES.PT_102:
            self.client.command("alter table F102Sym delete where 1=1")
        if table == PRE_TABLES.PT_SORS:
            self.client.command("alter table sors delete where 1=1")

    def drop_error_data(self):
        pass
   #     self.client.command("alter table app_row delete where okved=''")

    def db_get_frames_by_facetype(self, ft,having_zero_mass=True) -> pd.DataFrame:
        if having_zero_mass:
            qry_str = (f"select date_reg, workers,region,credits_mass from app_row where typeface={ft} order by"
                       f" date_reg,region")  # okved,
        else:
            qry_str = (
                f"select date_reg, workers,region,credits_mass from app_row where typeface={ft} and credits_mass>0"
                f" and workers>0 order by date_reg,region")  # okved,
        raw_data: pd.DataFrame = self.client.query_df(qry_str)
        return raw_data

    def db_ret_okato(self):
        okato_sql = "select * from regions order by region"
        okato_data = self.client.query_df(okato_sql)
        return okato_data

    def db_insert_data(self, df: pd.DataFrame):
        settings = {'async_insert': 1}
        self.client.insert_df(table='app_row', df=df, column_names=['date_reg', 'workers',# 'okved',
                                                                    'region', 'typeface',
                                                                    'credits_mass'],
                              column_type_names=['Date', 'Int32',  'Int32', 'Float32',
                                                 'Float32'], settings=settings)#'String',

    def db_get_minmax(self):
        return self.client.query_df(query="select min(date_reg) as min_date,max(date_reg) as max_date from app_row")

    def db_update_rows_kv(self, kvframe: pd.DataFrame):
        for item in kvframe.itertuples():
            date_reg = item[1]
            key_r = item[2]
            parameters = {'key_r': key_r,
                          #         'usd': usd,
                          #         'eur': eur,
                          'date_reg': date_reg}
            query = ("alter table app_row update ratekey={key_r:Float32} where "
                     "date_reg={date_reg:DateTime}")  # ,usd=={usd:Float32},eur={eur:Float32}
            self.client.command(query, parameters=parameters)

    def db_update_rows(self, frame_to_update, typeface):
        for item in frame_to_update.itertuples():
            date_reg = item[1]
            region = item[2]
            credits = item[3]
            face = typeface.value
            parameters = {
                'date_reg': date_reg,
                'okato': region,
                'credits': credits,
                'typeface': face
            }
            query = ("alter table app_row update credits_mass={credits:Float32} where "
                     "date_req={date_reg:DateTime} and region={okato:Int32} and typeface={typeface:Int32}")
            self.client.command(query, parameters)

    def db_fill_glossary(self, parser, mindate=dt.datetime.strptime('01.01.2010', '%d.%m.%Y'),
                         maxdate=dt.datetime.today()):
        asyncio.run(write_log(message=f'Load data from CBR:{dt.datetime.now()}', severity=SEVERITY.INFO))
        kv_dframe = parser.get_rates_cbr(mindate=mindate, maxdate=maxdate)
        self.db_prepare_tables(PRE_TABLES.PT_CBR)
        asyncio.run(
            write_log(message=f'Glossary:Write data to ClickHouse started:{dt.datetime.now()}', severity=SEVERITY.INFO))
        self.client.insert_df(table='app_cbr', df=kv_dframe, column_names=['date_', 'keyrate'],  # , 'usd', 'eur'],
                              column_type_names=['Date', 'Float32'])  # , 'Float32', 'Float32'])
        asyncio.run(write_log(message=f'Glossary:Finished:{dt.datetime.now()}', severity=SEVERITY.INFO))
        return kv_dframe

    def db_write_okato(self, frame):
        counts = self.client.query_df("select count(*) from okato")
        if (counts['count()'][0] == 0):
            self.client.insert_df("okato", frame, column_names=["okato_code", "regname"],
                                  column_type_names=["Int64", "String"])

    def db_write_sors(self, frame_okato, frame_sors):
        self.client.insert_df("sors", frame_sors,
                              column_names=["region", "total", "msp_total", "il_total", "date_rep", "okato"],
                              column_type_names=["String", "Float64", "Float64", "Float64", "Date", "Int32"])
        for item in frame_okato.itertuples():
            parameters = {'okval': item[1], 'reg': item[2]}
            query = ("alter table sors update okato={okval:Int64} where "
                     "region={reg:String}")
            self.client.command(query, parameters=parameters)
        unknwn = self.client.query_df("select distinct(region) from sors where okato=0")
        regions = list(map(lambda x: x.split(' '), unknwn['region'].tolist()))
        for _ in range(len(regions)):
            regions[_] = [x for x in regions[_] if len(x.strip()) > 2]
            if len(regions[_]) > 3:
                regions[_].reverse()
        for item in regions:
            filter_name = f"%{item[0].strip()}%"
            query = f"select okato_code from okato where regname like '{filter_name}'"
            result = self.client.query_df(query)
            query = f"select * from sors where region like '{filter_name}'"
            df = self.client.query_df(query)
            df['okato'] = result['okato_code'].tolist()[0]
            self.client.insert_df("sors", df,
                                  column_names=["region", "total", "msp_total", "il_total", "date_rep", "okato"],
                                  column_type_names=["String", "Float64", "Float64", "Float64", "Date", "Int32"])
        self.client.command("alter table sors delete where okato=0")

    def get_unq_dates(self, typeface):
        if typeface == MSP_CLASS.MSP_UL:
            query = f"select date_reg from serv_app_rows where typeface={typeface.value}"
        else:
            query = f"select date_req from serv_app_rows where typeface={typeface.value}"
        return self.client.query_df(query)

    def get_unq_okatos(self):
        return self.client.query_df("select * from serv_sors_regs order by okato_reg")

    def get_sors(self):
        return self.client.query_df("select * from sors order by date_rep")

    def get_app_reduced(self, typeface):
        return self.client.query(
            f"select * from serv_app_rows_reduced where typeface={typeface.value} order by date_reg")

    def update_app(self, frame:pd.DataFrame, typeface, processors_count):
        self.client.command(f"alter table app_row delete where typeface={typeface.value}")
        settings = {'async_insert': 1}
        self.client.insert_df("app_row", frame,
                              column_names=['date_reg', 'workers', #'okved',
                                            'region',
                                            'credits_mass','typeface'],
                              column_type_names=['Date', 'Int32',  'Int32',  'Float32',
                                                 'Int32'], settings=settings)#'String',
        pass

    def db_get_workers_limits(self,typeface):
        if typeface == MSP_CLASS.MSP_UL:
            query=f"select distinct(workers) as wrks from app_row where typeface={typeface.value} order by workers"
            df=self.client.query_df(query)
            a_min=df.iloc[1]
            a_max=df.iloc[-2]
            return [a_min[0],a_max[0]]
        else:
            return [0,1]

def create_updated(row):
    date_rep = row[0]
    values = row[1]
    okato = row[2]
    typeface=row[3]
    parameters = {
        'date_reg': date_rep,
        'okato': okato,
        'credits': values,
        'typeface': typeface
    }
    return parameters


def db_create_storage():
    pass



