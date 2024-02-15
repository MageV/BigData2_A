import asyncio
import datetime as dt
import gc
import multiprocessing

import joblib
from joblib import parallel_backend
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from apputils.log import write_log
from config.appconfig import *
from ml.ai_hyper import *
from providers.df import df_clean_for_ai
from providers.db_clickhouse import db_get_frames_by_facetype


def ai_clean(mean_over, msp_type: MSP_CLASS, no_compare=True):
    raw_data = db_get_frames_by_facetype(msp_type.value, mean_over)
    if not no_compare:
        raw_data = df_clean_for_ai(raw_data)
        raw_data.drop(['date_reg', 'workers'], axis=1, inplace=True)
    else:
        raw_data.drop(['date_reg'], axis=1, inplace=True)
    return raw_data


def ai_learn(mean_over, features=None, scaler=AI_SCALER.AI_NONE, models_class=AI_MODELS.AI_REGRESSORS,
             msp_class=MSP_CLASS.MSP_UL):
    if features is None:
        features = ['*']
    best_model = None
    models_results = {}
    raw_data = ai_clean(mean_over, msp_class, False)
    # построение исходных данных модели
    frame_cols = raw_data.columns.tolist()
    if not features.__contains__('*'):
        criteria_list = [x for x in frame_cols if x not in features]
    else:
        criteria_list = frame_cols
    pre_work_data = raw_data[criteria_list]
    df_X = pre_work_data.drop(['workers_ai'], axis=1)
    df_Y = pre_work_data['workers_ai'].values
    gc.collect()
    # Расчет моделей и выбор наилучшей
    with parallel_backend("multiprocessing", n_jobs=multiprocessing.cpu_count() - 2):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.3, shuffle=True)#, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
        if scaler == AI_SCALER.AI_STD:
            X_train_scaled = scaler.fit(X_train)
            X_test_scaled = scaler.fit(X_test)
        elif scaler == AI_SCALER.AI_STD_TRF:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        models_Beyes = [GaussianNB(), BernoulliNB()]
        models_Regressor = [LogisticRegression(), Lasso(), Ridge(),
         RandomForestRegressor(), DecisionTreeRegressor(), ExtraTreesRegressor()]
        models_ML = [SVR(),NuSVR(), LinearSVR()]
        models_trees = [RandomForestClassifier(), DecisionTreeClassifier(),ExtraTreesClassifier()]
        last_estimation = 0
        if models_class == AI_MODELS.AI_REGRESSORS:
            models = models_Regressor
        elif models_class == AI_MODELS.AI_BEYES:
            models = models_Beyes
        elif models_class == AI_MODELS.AI_ML:
            models = models_ML
        elif models_class == AI_MODELS.AI_ALL:
            models = models_Regressor + models_Beyes + models_ML + models_trees
        elif models_class == AI_MODELS.AI_TREES:
            models = models_trees
        for _ in range(len(models)):
            current_model = models[_]

            current_name = current_model.__repr__().split('(')[0].lower()
            asyncio.run(write_log(message=f'Model {models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))

            if current_name == 'elasticnet':
                search = HalvingGridSearchCV(current_model, param_elastic,
                                             scoring="r2", verbose=3, factor=5, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'randomforestregressor' or current_name == 'extratreesregressor':
                search = HalvingGridSearchCV(current_model, param_rf,
                                             scoring="r2", verbose=3, factor=5, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'lasso':
                search = HalvingGridSearchCV(current_model, param_lasso,
                                             scoring="r2", verbose=3, factor=5, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'ridge':
                search = HalvingGridSearchCV(current_model, param_ridge,
                                             scoring="r2", verbose=3, factor=5, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'histgradientboostingregressor':
                search = HalvingGridSearchCV(current_model, param_hbr,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name.__contains__('svr'):
                search = HalvingGridSearchCV(current_model, param_svc,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'logisticregression':
                search = HalvingGridSearchCV(current_model, param_lr,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'decisiontreeregressor':
                search = HalvingGridSearchCV(current_model, param_dtr,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'xgbregressor':
                search = HalvingGridSearchCV(current_model, param_xgboost,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'bernoullinb' or current_name == 'mutlinomialnb':
                search = HalvingGridSearchCV(current_model, param_gaussian_cat_multi,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search
            elif current_name == 'gaussiannb':
                search = HalvingGridSearchCV(current_model, param_gaussian_nb,
                                             scoring="r2", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search

            elif current_name.__contains__('classifier'):
                search = HalvingGridSearchCV(current_model, param_rfc,
                                             scoring="f1", verbose=3, factor=2, n_jobs=-1, min_resources=20,cv=10)
                current_model = search

            current_model.fit(X_train_scaled, Y_train)
            result = current_model.predict(X_test_scaled)
#            estimation_accuracy = metrics.accuracy_score(Y_test, result)
            if models_class==AI_MODELS.AI_TREES:
                estimation = metrics.f1_score(Y_test, result,average="micro")
            else:
                estimation=metrics.r2_score(Y_test,result)
            models_results.update(
                {str(current_model.best_estimator_).split('(')[0]: (estimation)})
            asyncio.run(
                write_log(message=f"Model:{str(current_model.best_estimator_).split('(')[0]}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"F1(R2):(good ~1,bad ~0) ,"
                                          f" {estimation}", severity=SEVERITY.INFO))
#            asyncio.run(write_log(message=f"ACCURACY: ,"
 #                                         f" {estimation_accuracy}", severity=SEVERITY.INFO))
            if best_model is None:
                best_model = current_model
                last_estimation = estimation
            elif estimation > last_estimation:
                best_model = current_model
                last_estimation = estimation
            current_model = None
            result = None
            gc.collect()
    asyncio.run(write_log(message=print(models_results), severity=SEVERITY.INFO))
    name = str(best_model.best_estimator_).split('(')[0]
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
