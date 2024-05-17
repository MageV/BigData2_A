import numpy as np
import pandas as pd

from apputils.utils import multiclass_binning, detect_distribution
from scipy import stats

from providers.db_clickhouse import *
import pandasql as ps


def df_recode_workers(df: pd.DataFrame):
    df_o = df
    cmpr = df_o['sworkers'].le(df_o['sworkers'].shift(1)).replace(True, 1).replace(False, 0)
    df_o['estimated'] = cmpr
    df_o.dropna(inplace=True)
    return df_o


def df_clean_for_ai(df: pd.DataFrame, dbprovider, msp_type, multiclass=False):
    okatos = dbprovider.db_ret_okato()
    df_o = pd.DataFrame()
    df_mm = dbprovider.db_get_workers_limits(msp_type)
    for item in okatos.itertuples():
        #       asyncio.run(write_log(message=f'OKATO:{item.region}', severity=SEVERITY.INFO))
        subset = df[df['region'] == item.region]
        df_1 = df_recode_workers(subset)
        df_o = pd.concat([df_o, df_1], axis=0, ignore_index=True)
    return df_o


def df_fill_credit_apps(typeface, dates_frame, sors_frame, app_frame, debt_frame):
    if typeface == MSP_CLASS.MSP_UL:
        work_frame = sors_frame[sors_frame["msp_total"] > 0]
        debt_work = debt_frame[debt_frame["msp_total"] > 0]
        work_frame = work_frame[['date_rep', 'okato', 'msp_total']]
        debt_work = debt_work[['date_rep', 'okato', 'msp_total']]
    else:
        work_frame = sors_frame[sors_frame["il_total"] > 0]
        work_frame = work_frame[['date_rep', 'okato', 'il_total']]
        debt_work = debt_frame[debt_frame["il_total"] > 0]
        debt_work = debt_work[['date_rep', 'okato', 'il_total']]

    min_date = dates_frame['min_date'].values[0]
    max_date = dates_frame['max_date'].values[0]
    work_frame = work_frame[(work_frame["date_rep"] >= min_date) & (work_frame["date_rep"] <= max_date)]
    for item in work_frame.itertuples():
        app_frame.loc[(app_frame['date_reg'] == item.date_rep) & (app_frame['region'] == item.okato), "credits_mass"] = \
            item[3]
    debt_work = debt_work[(debt_work["date_rep"] >= min_date) & (debt_work["date_rep"] <= max_date)]
    for item in debt_work.itertuples():
        app_frame.loc[(app_frame['date_reg'] == item.date_rep) & (app_frame['region'] == item.okato), "debt_mass"] = \
            item[3]
    app_frame["typeface"] = typeface.value
    app_frame.dropna(inplace=True)
    return app_frame


def df_clean(db_provider, appframe, msp_type: MSP_CLASS = MSP_CLASS.MSP_UL, is_multiclass=False, istf=False):
    if not is_multiclass:
        raw_data = appframe.loc[appframe['typeface'] == msp_type.value]
        raw_data = df_clean_for_ai(raw_data, db_provider, msp_type)
        raw_data.drop(['date_reg', 'sworkers'], axis=1, inplace=True)
        return raw_data
    # raw_data["facetype"] = msp_type.value
    else:
        raw_data = appframe.copy(deep=True)
        raw_data, boundaries, labels = multiclass_binning(raw_data, 'sworkers', classes=8)
        raw_data.drop(['date_reg', 'sworkers'], axis=1, inplace=True)
        return raw_data, boundaries, labels


def df_create_raw_data(db_provider, appframe, is_multiclass):
    if not is_multiclass:
        raw_data_1 = df_clean(db_provider, appframe, MSP_CLASS.MSP_UL, is_multiclass)
        raw_data_1.dropna(inplace=True)
        raw_data_2 = df_clean(db_provider, appframe, MSP_CLASS.MSP_FL, is_multiclass)
        raw_data_2.dropna(inplace=True)
        raw_data = pd.concat([raw_data_1, raw_data_2], axis=0, ignore_index=True)
        raw_data.dropna(inplace=True)
        return raw_data
    else:
        raw_data, boundaries, labels = df_clean(db_provider, appframe, is_multiclass=is_multiclass)
        raw_data.dropna(inplace=True)
        return raw_data, boundaries, labels


def df_remove_outliers(df, okatos, estims):
    returns_df = pd.DataFrame()
    if len(df) > 0:
        for item in okatos.itertuples():
            dfs = df.loc[df["region"] == item.okato_reg]
            mid = dfs[estims].mean()
            sigma = dfs[estims].std()
            out_ix = dfs[(dfs[estims] < mid - 3 * sigma) | (df[estims] > mid + 3 * sigma)].index
            out_ix = list(set(out_ix))
            dfs.loc[out_ix,'sworkers']=mid
            returns_df = pd.concat([returns_df, dfs], axis=0, ignore_index=True)
        return returns_df
    return None
