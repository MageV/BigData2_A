import asyncio
import datetime as dt
import multiprocessing
import joblib
import pandas as pd
from joblib import parallel_backend
from sklearn import metrics
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBRegressor

from apputils.log import write_log
from config.appconfig import *
from ml.ai_hyper import *


def ai_learn(mean_over, features=None, ):
    if features is None:
        features = ['*']
    best_model = None
    qry_str = "select date_reg, workers,region, typeface, ratekey,usd,eur from app_row order by date_reg"
    if mean_over == AI_FACTOR.AIF_KR:
        qry_str = qry_str.replace(',usd,eur', '')
    elif mean_over == AI_FACTOR.AIF_EUR:
        qry_str = qry_str.replace(',ratekey,usd', '')
    elif mean_over == AI_FACTOR.AIF_USD:
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
    with parallel_backend("multiprocessing",n_jobs=multiprocessing.cpu_count() - 2):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        models = [XGBRegressor(), Lasso(), Ridge(), ElasticNet(),
                  OneVsRestClassifier(LinearSVC(dual="auto", random_state=0)),
                  LogisticRegression()]
        last_estimation = 0
        for _ in range(len(models)):
            current_model = models[_]
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            asyncio.run(write_log(message=f'Model{models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))
            if current_model.__repr__().split('(')[0].lower() == 'ridge':
                search = GridSearchCV(current_model, param_ridge, scoring='neg_mean_absolute_percentage_error', cv=cv)
                current_model = search
            elif current_model.__repr__().split('(')[0].lower() == 'lasso':
                search = GridSearchCV(current_model, param_lasso, scoring='neg_mean_absolute_percentage_error', cv=cv)
                current_model = search
            elif current_model.__repr__().split('(')[0].lower() == 'elasticnet':
                search = GridSearchCV(current_model, param_elastic, scoring='neg_mean_absolute_percentage_error', cv=cv)
                current_model = search
            elif current_model.__repr__().split('(')[0].lower() == 'linearregression':
                search = GridSearchCV(current_model, param_lr, scoring='neg_mean_absolute_percentage_error', cv=cv)
                current_model = search
            current_model.fit(X_train_scaled, Y_train)
            result = current_model.predict(X_test_scaled)
            estimation_mae = mean_absolute_percentage_error(Y_test, result)
            estimation_r2 = metrics.r2_score(Y_test, result)
            asyncio.run(write_log(message=f"R2:(good ~1,bad ~0) ,"
                                          f" {estimation_r2}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"MAPE(%): ,"
                                          f" {estimation_mae}", severity=SEVERITY.INFO))
            if best_model == None:
                best_model = current_model
                last_estimation = estimation_r2
            elif estimation_r2 > last_estimation:
                best_model = current_model
                last_estimation = estimation_r2

    name = best_model.__repr__().split('(')[0].lower()
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
