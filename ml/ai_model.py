import gc
import json
import multiprocessing
import pickle

import joblib
from joblib import parallel_backend
from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, ElasticNet, ElasticNetCV, SGDRegressor, \
    PassiveAggressiveClassifier, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.svm import SVR, NuSVR, LinearSVR, SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.ai_hyper import *
from providers.db_clickhouse import *
from providers.df import df_clean_for_ai


def ai_clean(db_provider, msp_type: MSP_CLASS, classifiers=True):
    raw_data = db_provider.db_get_frames_by_facetype(msp_type.value, )
    if classifiers:
        raw_data = df_clean_for_ai(raw_data, db_provider)
        raw_data.drop(['date_reg', 'workers'], axis=1, inplace=True)
    else:
        raw_data.drop(['date_reg'], axis=1, inplace=True)
    raw_data["facetype"] = msp_type.value
    return raw_data


def ai_learn(db_provider, features=None, scaler=AI_SCALER.AI_NONE, models_class=AI_MODELS.AI_REGRESSORS):
    if features is None:
        features = ['*']
    best_model = None
    models_results = {}
    raw_data_1 = ai_clean(db_provider, MSP_CLASS.MSP_UL)
    raw_data_2 = ai_clean(db_provider, MSP_CLASS.MSP_FL)
    raw_data = pd.concat([raw_data_1, raw_data_2], axis=0, ignore_index=True)
    # построение исходных данных модели
    frame_cols = raw_data.columns.tolist()
    if not features.__contains__('*'):
        criteria_list = [x for x in frame_cols if x not in features]
    else:
        criteria_list = frame_cols
    pre_work_data = raw_data[criteria_list]
    label_encoder = LabelEncoder()
    pre_work_data.loc[:, 'okved'] = label_encoder.fit_transform(pre_work_data.loc[:, 'okved'])
    df_X = pre_work_data.drop(['workers_ai'], axis=1)
    df_Y = pre_work_data['workers_ai'].values
    gc.collect()
    cv_rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Расчет моделей и выбор наилучшей
    with (parallel_backend("multiprocessing", n_jobs=multiprocessing.cpu_count() - 2)):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.25,
                                                            shuffle=True, random_state=42)
        scaler = RobustScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
        if scaler == AI_SCALER.AI_STD:
            X_train_scaled = scaler.fit(X_train)
            X_test_scaled = scaler.fit(X_test)
        elif scaler == AI_SCALER.AI_STD_TRF:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)
        models_Beyes = [GaussianNB(), BernoulliNB()]
        net_models = [ElasticNetCV(), ElasticNet(), ]
        models_Regressor = [BaggingRegressor(),
                            LinearRegression(),
                            SGDRegressor(),
                            AdaBoostRegressor(),
                            Lasso(), Ridge(),
                            RandomForestRegressor(), DecisionTreeRegressor(), ExtraTreesRegressor(), ]
        models_ML = [SVR(), NuSVR(), LinearSVR(), SVC(), NuSVC(), LinearSVC()]
        models_classifiers = [LogisticRegression(), LogisticRegressionCV(), PassiveAggressiveClassifier(),
                              MLPClassifier(), AdaBoostClassifier(),
                              RandomForestClassifier(),
                              DecisionTreeClassifier(), ExtraTreesClassifier()]
        models_discsriminant = [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]
        # NO MEMORY FOR
        experimental_models = [GradientBoostingClassifier(), GaussianProcessClassifier(),
                               HistGradientBoostingRegressor()]
        last_estimation = 0
        if models_class == AI_MODELS.AI_REGRESSORS:
            models = models_Regressor
        elif models_class == AI_MODELS.AI_BEYES:
            models = models_Beyes
        elif models_class == AI_MODELS.AI_ML:
            models = models_ML
        elif models_class == AI_MODELS.AI_ALL:
            models = models_classifiers + models_Regressor + models_Beyes  # + models_ML
        elif models_class == AI_MODELS.AI_CLASSIFIERS:
            models = models_classifiers
        elif models_class == AI_MODELS.AI_EXPERIMENTAL:
            models = experimental_models
        for _ in range(len(models)):
            current_model = models[_]
            current_name = current_model.__repr__().split('(')[0].lower()
            asyncio.run(write_log(message=f'Model {models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))
            if current_name == 'elasticnetcv':
                search = HalvingGridSearchCV(current_model, param_elastic_cv,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            if current_name == 'sgdregressor':
                search = HalvingGridSearchCV(current_model, param_sgd_regr,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search

            if current_name == 'elasticnet':
                search = HalvingGridSearchCV(current_model, param_elastic,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name in ['randomforestregressor', 'extratreesregressor']:
                search = HalvingGridSearchCV(current_model, param_rf,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'lasso':
                search = HalvingGridSearchCV(current_model, param_lasso,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'ridge':
                search = HalvingGridSearchCV(current_model, param_ridge,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'histgradientboostingregressor':
                search = HalvingGridSearchCV(current_model, param_hbr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name.__contains__('svr'):
                search = HalvingGridSearchCV(current_model, param_svc,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'logisticregression':
                search = HalvingGridSearchCV(current_model, param_lr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'decisiontreeregressor':
                search = HalvingGridSearchCV(current_model, param_dtr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'xgbregressor':
                search = HalvingGridSearchCV(current_model, param_xgboost,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'bernoullinb' or current_name == 'mutlinomialnb':
                search = HalvingGridSearchCV(current_model, param_gaussian_cat_multi,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'gaussiannb':
                search = HalvingGridSearchCV(current_model, param_gaussian_nb,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search

            elif current_name in ['randomforestclassifier', 'decisiontreeclassifier', 'extratreesclassifier']:
                search = HalvingGridSearchCV(current_model, param_rfc,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=cv_rsk)
                current_model = search

            elif current_name.__contains__('adaboostclassifier'):
                search = HalvingGridSearchCV(current_model, param_ada_classifier,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=cv_rsk)
                current_model = search

            elif current_name.__contains__('adaboostregressor'):
                search = HalvingGridSearchCV(current_model, param_ada_regressor,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('mlpclassifier'):
                search = HalvingGridSearchCV(current_model, param_mlp,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('gaussianprocessclassifier'):
                search = HalvingGridSearchCV(current_model, param_gauss_proc,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('gradientboostingclassifier'):
                search = HalvingGridSearchCV(current_model, param_grad_boost,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__("baggingregressor"):
                search = HalvingGridSearchCV(current_model, param_bagging_regr,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search
            elif current_name.__contains__("linearregression"):
                search = HalvingGridSearchCV(current_model, param_linear_regr,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search
            elif current_name.__contains__("passiveaggressiveclassifier"):
                search = HalvingGridSearchCV(current_model, param_pass_agg_clf,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__("logisticregressioncv"):
                search = HalvingGridSearchCV(current_model, param_logr_cv,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            estimator = current_model.fit(X_train_scaled, Y_train)
            result = current_model.best_estimator_.predict(X_test_scaled)
            if models_class == AI_MODELS.AI_CLASSIFIERS:
                estimation_accuracy = metrics.accuracy_score(Y_test, result)
            else:
                estimation_accuracy = metrics.r2_score(Y_test, result)
            #            estimation_accuracy = metrics.accuracy_score(Y_test, result)
            #            if models_class == AI_MODELS.AI_TREES:
            #            estimation = metrics.r2_score(Y_test, result, multioutput="uniform_average")
            #            else:
            #                estimation = metrics.mean_absolute_error(Y_test, result)
            modelname = ""
            #           if current_model.__str__().__contains__("SVC") or current_model.__str__().__contains__(
            #                   "SGDR") or current_model.__str__().__contains__("NetCV") or current_model.__str__().__contains__("Bagging"):
            #               modelname = current_model.estimator_
            #           else:
            modelname = str(current_model.best_estimator_).split('(')[0]
            score = current_model.best_score_
            models_results.update(
                {modelname: (score, estimation_accuracy)})
            asyncio.run(
                write_log(message=f"Model:{modelname}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"Good(>0.8):Medium(>0.6)"
                                          f" {estimation_accuracy}", severity=SEVERITY.INFO))
            #            asyncio.run(write_log(message=f"ACCURACY: ,"
            #                                         f" {estimation_accuracy}", severity=SEVERITY.INFO))
            if best_model is None:
                best_model = current_model
                last_estimation = estimation_accuracy
            elif estimation_accuracy > last_estimation:
                best_model = current_model
                last_estimation = estimation_accuracy
            current_model = None
            result = None
            gc.collect()
    asyncio.run(write_log(message=print(models_results), severity=SEVERITY.INFO))
    name = str(best_model.best_estimator_).split('(')[0]
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
        with open(f"{MODEL_STORE}{name}_score.txt", "w+") as file:
            file.writelines(f"score:{last_estimation}")
        with open(f"{MODEL_STORE}{name}_parameters.json", "w+") as file:
            json.dump(best_model.best_params_, file)
        with open(f'{MODEL_STORE}label_encoder{models_class.value}.pickle', 'wb') as file:
            pickle.dump(label_encoder, file, pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))


def ai_learn_v2(db_provider, features=None, scaler=AI_SCALER.AI_NONE, models_class=AI_MODELS.AI_REGRESSORS):
    if features is None:
        features = ['*']
    best_model = None
    models_results = {}
    if models_class.value == '_classifiers_':
        raw_data_1 = ai_clean(db_provider, MSP_CLASS.MSP_UL)
        raw_data_2 = ai_clean(db_provider, MSP_CLASS.MSP_FL)
    else:
        raw_data_1 = ai_clean(db_provider, MSP_CLASS.MSP_UL, False)
        raw_data_2 = ai_clean(db_provider, MSP_CLASS.MSP_FL, False)
    raw_data = pd.concat([raw_data_1, raw_data_2], axis=0, ignore_index=True)
    # построение исходных данных модели
    frame_cols = raw_data.columns.tolist()
    if not features.__contains__('*'):
        criteria_list = [x for x in frame_cols if x not in features]
    else:
        criteria_list = frame_cols
    pre_work_data = raw_data[criteria_list]
    #    label_encoder = LabelEncoder()
    #    pre_work_data.loc[:, 'okved'] = label_encoder.fit_transform(pre_work_data.loc[:, 'okved'])
    df_X = pre_work_data.drop(['workers_ai'], axis=1)
    df_Y = pre_work_data['workers_ai'].values
    gc.collect()
    cv_rsk = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # Расчет моделей и выбор наилучшей
    with (parallel_backend("multiprocessing", n_jobs=multiprocessing.cpu_count() - 2)):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.25,
                                                            shuffle=True, random_state=42)
        scaler = RobustScaler()
        X_train_scaled = X_train
        X_test_scaled = X_test
        if scaler == AI_SCALER.AI_STD:
            X_train_scaled = scaler.fit(X_train)
            X_test_scaled = scaler.fit(X_test)
        elif scaler == AI_SCALER.AI_STD_TRF:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.fit_transform(X_test)
        models_Beyes = [GaussianNB(), BernoulliNB()]
        net_models = [ElasticNetCV(), ElasticNet(), ]
        models_Regressor = [BaggingRegressor(),
                            LinearRegression(),
                            SGDRegressor(),
                            AdaBoostRegressor(),
                            Lasso(), Ridge(),
                            RandomForestRegressor(), DecisionTreeRegressor(), ExtraTreesRegressor(), ]
        models_ML = [SVR(), NuSVR(), LinearSVR(), SVC(), NuSVC(), LinearSVC()]
        models_classifiers = [LogisticRegression(), LogisticRegressionCV(), PassiveAggressiveClassifier(),
                              MLPClassifier(), AdaBoostClassifier(),
                              RandomForestClassifier(),
                              DecisionTreeClassifier(), ExtraTreesClassifier()]
        models_discsriminant = [LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]
        # NO MEMORY FOR
        experimental_models = [GradientBoostingClassifier(), GaussianProcessClassifier(),
                               HistGradientBoostingRegressor()]
        last_estimation = 0
        if models_class == AI_MODELS.AI_REGRESSORS:
            models = models_Regressor
        elif models_class == AI_MODELS.AI_BEYES:
            models = models_Beyes
        elif models_class == AI_MODELS.AI_ML:
            models = models_ML
        elif models_class == AI_MODELS.AI_ALL:
            models = models_classifiers + models_Regressor + models_Beyes  # + models_ML
        elif models_class == AI_MODELS.AI_CLASSIFIERS:
            models = models_classifiers
        elif models_class == AI_MODELS.AI_EXPERIMENTAL:
            models = experimental_models
        for _ in range(len(models)):
            current_model = models[_]
            current_name = current_model.__repr__().split('(')[0].lower()
            asyncio.run(write_log(message=f'Model {models[_]} '
                                          f'started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))
            if current_name == 'elasticnetcv':
                search = HalvingGridSearchCV(current_model, param_elastic_cv,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            if current_name == 'sgdregressor':
                search = HalvingGridSearchCV(current_model, param_sgd_regr,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search

            if current_name == 'elasticnet':
                search = HalvingGridSearchCV(current_model, param_elastic,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name in ['randomforestregressor', 'extratreesregressor']:
                search = HalvingGridSearchCV(current_model, param_rf,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'lasso':
                search = HalvingGridSearchCV(current_model, param_lasso,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'ridge':
                search = HalvingGridSearchCV(current_model, param_ridge,
                                             verbose=0, factor=5, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'histgradientboostingregressor':
                search = HalvingGridSearchCV(current_model, param_hbr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name.__contains__('svr'):
                search = HalvingGridSearchCV(current_model, param_svc,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'logisticregression':
                search = HalvingGridSearchCV(current_model, param_lr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'decisiontreeregressor':
                search = HalvingGridSearchCV(current_model, param_dtr,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'xgbregressor':
                search = HalvingGridSearchCV(current_model, param_xgboost,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'bernoullinb' or current_name == 'mutlinomialnb':
                search = HalvingGridSearchCV(current_model, param_gaussian_cat_multi,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search
            elif current_name == 'gaussiannb':
                search = HalvingGridSearchCV(current_model, param_gaussian_nb,
                                             verbose=0, factor=2, n_jobs=-1,
                                             min_resources=20, cv=cv_rsk)
                current_model = search

            elif current_name in ['randomforestclassifier', 'decisiontreeclassifier', 'extratreesclassifier']:
                search = HalvingGridSearchCV(current_model, param_rfc,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=cv_rsk)
                current_model = search

            elif current_name.__contains__('adaboostclassifier'):
                search = HalvingGridSearchCV(current_model, param_ada_classifier,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=cv_rsk)
                current_model = search

            elif current_name.__contains__('adaboostregressor'):
                search = HalvingGridSearchCV(current_model, param_ada_regressor,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('mlpclassifier'):
                search = HalvingGridSearchCV(current_model, param_mlp,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('gaussianprocessclassifier'):
                search = HalvingGridSearchCV(current_model, param_gauss_proc,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__('gradientboostingclassifier'):
                search = HalvingGridSearchCV(current_model, param_grad_boost,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__("baggingregressor"):
                search = HalvingGridSearchCV(current_model, param_bagging_regr,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search
            elif current_name.__contains__("linearregression"):
                search = HalvingGridSearchCV(current_model, param_linear_regr,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search
            elif current_name.__contains__("passiveaggressiveclassifier"):
                search = HalvingGridSearchCV(current_model, param_pass_agg_clf,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            elif current_name.__contains__("logisticregressioncv"):
                search = HalvingGridSearchCV(current_model, param_logr_cv,
                                             verbose=0, factor=2, n_jobs=-1, min_resources=20,
                                             cv=5)
                current_model = search

            estimator = current_model.fit(X_train_scaled, Y_train)
            result = current_model.best_estimator_.predict(X_test_scaled)
            if models_class == AI_MODELS.AI_CLASSIFIERS:
                estimation_accuracy = metrics.accuracy_score(Y_test, result)
            else:
                estimation_accuracy = metrics.r2_score(Y_test, result)
            #            estimation_accuracy = metrics.accuracy_score(Y_test, result)
            #            if models_class == AI_MODELS.AI_TREES:
            #            estimation = metrics.r2_score(Y_test, result, multioutput="uniform_average")
            #            else:
            #                estimation = metrics.mean_absolute_error(Y_test, result)
            modelname = ""
            #           if current_model.__str__().__contains__("SVC") or current_model.__str__().__contains__(
            #                   "SGDR") or current_model.__str__().__contains__("NetCV") or current_model.__str__().__contains__("Bagging"):
            #               modelname = current_model.estimator_
            #           else:
            modelname = str(current_model.best_estimator_).split('(')[0]
            score = current_model.best_score_
            models_results.update(
                {modelname: (score, estimation_accuracy)})
            asyncio.run(
                write_log(message=f"Model:{modelname}", severity=SEVERITY.INFO))
            asyncio.run(write_log(message=f"Good(>0.8):Medium(>0.6)"
                                          f" {estimation_accuracy}", severity=SEVERITY.INFO))
            #            asyncio.run(write_log(message=f"ACCURACY: ,"
            #                                         f" {estimation_accuracy}", severity=SEVERITY.INFO))
            if best_model is None:
                best_model = current_model
                last_estimation = estimation_accuracy
            elif estimation_accuracy > last_estimation:
                best_model = current_model
                last_estimation = estimation_accuracy
            current_model = None
            result = None
            gc.collect()
    asyncio.run(write_log(message=print(models_results), severity=SEVERITY.INFO))
    name = str(best_model.best_estimator_).split('(')[0]
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
        with open(f"{MODEL_STORE}{name}_score.txt", "w+") as file:
            file.writelines(f"score:{last_estimation}")
        with open(f"{MODEL_STORE}{name}_parameters.json", "w+") as file:
            json.dump(best_model.best_params_, file)
    #        with open(f'{MODEL_STORE}label_encoder{models_class.value}.pickle', 'wb') as file:
    #            pickle.dump(label_encoder, file, pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
