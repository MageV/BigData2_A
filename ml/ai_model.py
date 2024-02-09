import asyncio
import datetime as dt
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from apputils.archivers import ArchiveManager
from apputils.log import write_log
from config.appconfig import *


def ai_learn(mean_over,features=None,):
    if features is None:
        features = ['*']
    best_model = None
    qry_str="select date_reg, workers,region, typeface, ratekey,usd,eur from app_row order by date_reg"
    if mean_over==AI_FACTOR.AIF_KR:
        qry_str=qry_str.replace(',usd,eur','')
    elif mean_over==AI_FACTOR.AIF_EUR:
        qry_str=qry_str.replace('ratekey,usd','')
    elif mean_over==AI_FACTOR.AIF_USD:
        qry_str = qry_str.replace('ratekey,', '')
        qry_str = qry_str.replace(',eur', '')
    raw_data: pd.DataFrame = click_client.query_df(qry_str)
    raw_data['year'] = raw_data['date_reg'].dt.year
    raw_data.drop('date_reg', axis=1, inplace=True)
  # построение исходных данных модели
    frame_cols = raw_data.columns.tolist()
    if not features.__contains__('*'):
        criteria_list = [x for x in frame_cols if x not in features]
    else:
        criteria_list = frame_cols
    pre_work_data = raw_data[criteria_list]
    df_X = pre_work_data.drop(['year', 'workers'], axis=1)
    df_Y = pre_work_data['workers'].values
    # Расчет моделей и выбор наилучшей
    with parallel_backend("multiprocessing"):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]
        last_estimation = sys.maxsize
        for _ in range(len(models)):
            asyncio.run(write_log(message=f'Model{models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))
            models[_].fit(X_train_scaled, Y_train)
            result = models[_].predict(X_test_scaled)
            estimation = mean_absolute_error(Y_test, result)
            if estimation < last_estimation:
                best_model = models[_]
                last_estimation = estimation
            asyncio.run(write_log(message=f"MAE: ,"
                                          f" {estimation}", severity=SEVERITY.INFO))
    name = best_model.__repr__().split('(')[0].lower()
    try:
        joblib.dump(best_model,f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
