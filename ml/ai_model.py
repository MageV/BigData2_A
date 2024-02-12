import asyncio
import datetime as dt
import gc
import multiprocessing

import joblib
import pandas as pd
from joblib import parallel_backend
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, RepeatedKFold, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from apputils.log import write_log
from config.appconfig import *
from ml.ai_hyper import *


def ai_learn(mean_over, features=None, scaler=AI_SCALER.AI_NONE):
    if features is None:
        features = ['*']
    best_model = None
    models_results = {}
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
    with parallel_backend("multiprocessing", n_jobs=multiprocessing.cpu_count() - 2):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.15)
        if scaler == AI_SCALER.AI_STD:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        elif scaler == AI_SCALER.AI_NONE:
            X_train_scaled = X_train
            X_test_scaled = X_test
        models = [ElasticNet(), Lasso(), Ridge(),
                  RandomForestRegressor(), DecisionTreeRegressor(),
                  ExtraTreesRegressor(), SVC(gamma='scale'), LogisticRegression()]
        last_estimation = 0
        for _ in range(len(models)):
            current_model = models[_]

            current_name = current_model.__repr__().split('(')[0].lower()
            asyncio.run(write_log(message=f'Model {models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))

            if current_name == 'elasticnet':
                search = HalvingGridSearchCV(current_model, param_elastic, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'randomforestregressor' or current_name == 'extratreesregressor':
                search = HalvingGridSearchCV(current_model, param_rf, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'lasso':
                search = HalvingGridSearchCV(current_model, param_lasso, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'ridge':
                search = HalvingGridSearchCV(current_model, param_ridge, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'histgradientboostingregressor':
                search = HalvingGridSearchCV(current_model, param_hbr, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'svc':
                search = HalvingGridSearchCV(current_model, param_svc, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'logisticregression':
                search = HalvingGridSearchCV(current_model, param_lr, cv=5,
                                             scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search
            elif current_name == 'decisiontreeregressor':
                search = HalvingGridSearchCV(current_model, param_dtr, cv=5,
                                            scoring='r2', verbose=3, factor=2, n_jobs=-1, min_resources=20)
                current_model = search

            current_model.fit(X_train_scaled, Y_train)
            result = current_model.predict(X_test_scaled)
            estimation_mae = mean_absolute_percentage_error(Y_test, result)
            estimation_r2 = metrics.r2_score(Y_test, result)
            models_results.update({current_model.__repr__().split('(')[0].lower(): (estimation_mae, estimation_r2)})
            asyncio.run(write_log(message=f"Model:{str(current_model.best_estimator_).split('(')[0]}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"R2:(good ~1,bad ~0) ,"
                                          f" {estimation_r2}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"MAPE(%): ,"
                                          f" {estimation_mae}", severity=SEVERITY.INFO))
            if best_model is None:
                best_model = current_model
                last_estimation = estimation_r2
            elif estimation_r2 > last_estimation:
                best_model = current_model
                last_estimation = estimation_r2
            current_model = None
            result = None
            gc.collect()
    asyncio.run(write_log(message=print(models_results), severity=SEVERITY.INFO))
    name = str(best_model.best_estimator_).split('(')[0]
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
