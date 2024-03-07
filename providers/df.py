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
    df_o = df
    cmpr = df_o['workers'].le(df_o['workers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['workers_ai'] = cmpr.reset_index()['workers']
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame, dbprovider):
    okatos = dbprovider.db_ret_okato()
    df_o = pd.DataFrame()
    for item in okatos.itertuples():
        asyncio.run(write_log(message=f'OKATO:{item[1]}', severity=SEVERITY.INFO))
        subset = df[df['region'] == item[1]]
        df_1 = df_recode_workers(subset)
        df_o = pd.concat([df_o, df_1], axis=0, ignore_index=True)
    return df_o


"""
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
    work_frame.replace({np.nan: None}, inplace=True)
    work_frame['symb_value'] = work_frame['symb_value'].astype('Float32')
    dt_index = pd.DatetimeIndex(work_frame['date_form'].values)
    work_frame['symb_value'] = work_frame['symb_value'].convert_dtypes(convert_floating=True)
    work_frame.drop('date_form', axis=1, inplace=True)
    work_frame['rowidx'] = dt_index
    work_frame.reset_index(inplace=True)
    work_frame.set_index('rowidx', inplace=True)
    work_frame.interpolate(method='time', inplace=True)
    work_frame.fillna(0, inplace=True)
    pass


def df_interpolate_over_typeface(typeface, dates_frame, okatos_frame, sors_frame, app_frame):
    loc_frame = app_frame
    if typeface == MSP_CLASS.MSP_UL:
        work_frame = sors_frame[['date_rep', 'okato', 'msp_total']]
    else:
        work_frame = sors_frame[['date_rep', 'okato', 'il_total']]
    min_date = dates_frame.min()[0]
    max_date = dates_frame.max()[0]
    work_frame = work_frame[(work_frame["date_rep"] >= min_date) & (work_frame["date_rep"] <= max_date)]
    for date_item in dates_frame.itertuples():
        date_parsed_item = dt_parser.parse(date_item[1].__str__())
        for okato_item in okatos_frame.itertuples():
            if len(work_frame[
                       (work_frame['date_rep'] == date_parsed_item) & (work_frame['okato'] == okato_item[1])]) == 0:
                work_frame.loc[-1] = [date_parsed_item, okato_item[1], np.nan]
                work_frame.index += 1
    # work_frame.replace({np.nan: None}, inplace=True)
    dt_index = pd.DatetimeIndex(work_frame['date_rep'].values)
    work_frame.drop('date_rep', axis=1, inplace=True)
    work_frame['rowidx'] = dt_index
    work_frame.reset_index(inplace=True)
    work_frame.set_index('rowidx', inplace=True)
    result_frame = pd.DataFrame()
    for item in okatos_frame.itertuples():
        okato = item[1]
        asyncio.run(write_log(message=f'Interpolation:{okato}', severity=SEVERITY.INFO))
        subframe = (work_frame[work_frame["okato"] == okato]).sort_index()
        subframe.drop('okato', axis=1, inplace=True)
        subframe.interpolate(method='time', inplace=True)
        subframe.fillna(0, inplace=True)
        subframe['okato'] = okato
        result_frame = pd.concat([result_frame, subframe], axis=0, ignore_index=False)
        asyncio.run(write_log(message=f'Interpolation:{okato} done', severity=SEVERITY.INFO))
    loc_frame["credits_mass"] = 0.0
    loc_frame["typeface"] = typeface.value
    for item in result_frame.itertuples():
        loc_frame.loc[(loc_frame['date_reg'] == item[0]) & (loc_frame["region"] == item[3]), "credits_mass"] = item[2]
    return loc_frame

"""


def df_fill_sors_apps(typeface, dates_frame, okatos_frame, sors_frame, app_frame):
    if typeface == MSP_CLASS.MSP_UL:
        work_frame = sors_frame[['date_rep', 'okato', 'msp_total']]
    else:
        work_frame = sors_frame[['date_rep', 'okato', 'il_total']]
    min_date = dates_frame['min_date'].values[0]
    max_date = dates_frame['max_date'].values[0]
    work_frame = work_frame[(work_frame["date_rep"] >= min_date) & (work_frame["date_rep"] <= max_date)]
    for item in work_frame.itertuples():
        app_frame.loc[(app_frame['date_reg'] == item[1]) & (app_frame['region'] == item[2]), "credits_mass"] = item[3]
        pass
    app_frame["typeface"]=typeface.value
    return app_frame
