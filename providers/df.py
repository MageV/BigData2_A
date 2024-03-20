import pandas as pd

from providers.db_clickhouse import *
import pandasql as ps


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
    app_frame["typeface"] = typeface.value
    return app_frame


