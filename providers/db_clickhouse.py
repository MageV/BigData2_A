import asyncio

import clickhouse_connect
import pandas as pd
import datetime as dt

from apputils.log import write_log
from config.appconfig import AI_FACTOR, SEVERITY, PRE_TABLES


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

    def db_get_frames_by_facetype(self, ft, mean_over) -> pd.DataFrame:
        qry_str = f"select date_reg, workers,region, ratekey,usd,eur from app_row where typeface={ft} order by date_reg,region"
        if mean_over == AI_FACTOR.AIF_KR:
            qry_str = qry_str.replace(',usd,eur', '')
        elif mean_over == AI_FACTOR.AIF_EUR:
            qry_str = qry_str.replace(',ratekey,usd', '')
        elif mean_over == AI_FACTOR.AIF_USD:
            qry_str = qry_str.replace('ratekey,', '')
            qry_str = qry_str.replace(',eur', '')
        raw_data: pd.DataFrame = self.client.query_df(qry_str)
        return raw_data

    def db_ret_okato(self):
        okato_sql = "select * from regions order by region"
        okato_data = self.client.query_df(okato_sql)
        return okato_data

    def db_insert_data(self, df: pd.DataFrame):
        settings = {'async_insert': 1}
        self.client.insert_df(table='app_row', df=df, column_names=['date_reg', 'workers', 'okved',
                                                                    'region', 'typeface', 'ratekey',
                                                                    'usd',
                                                                    'eur'],
                              column_type_names=['Date', 'Int32', 'String', 'Int32', 'Int32', 'Float32',
                                                 'Float32', 'Float32'], settings=settings)

    def db_get_minmax(self):
        return self.client.query(query="select min(date_reg),max(date_reg) from app_row")

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

    def db_fill_f102(self, frame):
        self.db_prepare_tables(PRE_TABLES.PT_102)
        asyncio.run(write_log(message=f'Write f102 symbols:{dt.datetime.now()}', severity=SEVERITY.INFO))
        self.client.insert_df(table='F102Sym', df=frame, column_names=["date_form", "symbol", "symb_value"],
                              column_type_names=["Date", "Int32", "Float32"])
        asyncio.run(write_log(message=f'Write f102 symbols:{dt.datetime.now()} finished', severity=SEVERITY.INFO))
        pass

    def db_get_f102(self, code):
        query_df = f"select date_form,symb_value from F102Sym where symbol={code}"
        return self.client.query_df(query=query_df)

    def db_get_dates(self):
        query_df = f"select distinct(date_reg) from app_row order by date_reg"
        return self.client.query_df(query=query_df)

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
        pass
