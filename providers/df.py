import asyncio

import numpy as np
import pandas

import pandas as pd
from dateutil import parser as dt_parser

from apputils.log import write_log
from config.appconfig import *
from config.appconfig import SEVERITY
from providers.db_clickhouse import *


def df_recode_workers(df: pd.DataFrame):
    df_o=df
    cmpr = df_o['workers'].le(df_o['workers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['workers_ai'] = cmpr.reset_index()['workers']
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame,dbprovider):
    okatos = dbprovider.db_ret_okato()
    df_o=pd.DataFrame()
    for item in okatos.itertuples():
        asyncio.run(write_log(message=f'OKATO:{item[1]}', severity=SEVERITY.INFO))
        subset=df[df['region']==item[1]]
        df_1 = df_recode_workers(subset)
        df_o = pd.concat([df_o,df_1], axis=0, ignore_index=True)
    return df_o


def df_prepare_f102(frame: pd.DataFrame, dates_frame: pd.DataFrame):
    work_frame = frame
    # date_list = dates_frame['date_reg'].tolist()
    for idx, item in dates_frame.itertuples():
        parsed_item = dt_parser.parse(item.__str__())
        asyncio.run(write_log(message=f'checked:{parsed_item}', severity=SEVERITY.INFO))
        if len(work_frame[work_frame['date_form'] == parsed_item]) == 0:
            work_frame.loc[-1] = [parsed_item, np.nan]
            work_frame.index += 1
    work_frame.sort_values(by='date_form', inplace=True)
    work_frame.replace({np.nan: None}, inplace = True)
    work_frame['symb_value']=work_frame['symb_value'].astype('Float32')
    dt_index=pd.DatetimeIndex(work_frame['date_form'].values)
    work_frame['symb_value']=work_frame['symb_value'].convert_dtypes(convert_floating=True)
    work_frame.drop('date_form',axis=1,inplace=True)
    work_frame['rowidx']=dt_index
    work_frame.reset_index(inplace=True)
    work_frame.set_index('rowidx',inplace=True)
    work_frame.interpolate(method='time',inplace=True)
    work_frame.fillna(0,inplace=True)
    pass
