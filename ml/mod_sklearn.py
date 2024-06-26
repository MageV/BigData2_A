import gc
import json
import multiprocessing
import joblib
from sklearnex import patch_sklearn

from apputils.T4 import test_correllation

patch_sklearn()
from joblib import parallel_backend
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, AdaBoostRegressor, IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, ElasticNet, ElasticNetCV, SGDRegressor, \
    PassiveAggressiveClassifier, LinearRegression, LogisticRegressionCV, RidgeClassifier, HuberRegressor, \
    Perceptron
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer, \
    PowerTransformer, SplineTransformer
from sklearn.svm import SVR, NuSVR, LinearSVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ml.sklearn_hyper import *
from providers.db_clickhouse import *
from providers.df import df_clean


def sk_learn_model(dataset, features=None, models_class=AI_MODELS.AI_REGRESSORS, is_multiclass=False):
    if features is None:
        features = ['*']
    best_model = None
    boundaries = None
    labels = None
    best_scaler = None
    models_results = {}
    if not is_multiclass:
        raw_data = dataset.copy(deep=True)
    else:
        raw_data = dataset[0].copy(deep=True)
        boundaries = dataset[1]
        labels = dataset[2]
        raw_data.dropna(inplace=True)
    #raw_data=raw_data.dropna(inplace=True)
    # построение исходных данных модели
    #raw_data['dead_credits'] = raw_data['debt_mass'] / raw_data['credits_mass']
    #raw_data = raw_data.drop(['debt_mass', 'credits_mass'], axis=1)
    frame_cols = raw_data.columns.tolist()

    if not features.__contains__('*'):
        criteria_list = [x for x in frame_cols if x not in features]
    else:
        criteria_list = frame_cols
    pre_work_data = raw_data[criteria_list]
    df_X = pre_work_data.drop(['estimated'], axis=1)
    df_Y = pre_work_data['estimated'].values
    cv_rsk = 5
    # Расчет моделей и выбор наилучшей
    with (parallel_backend("multiprocessing", n_jobs=multiprocessing.cpu_count() - 2)):
        X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.30,
                                                            shuffle=True, random_state=42, stratify=df_Y)

        class_classifiers = False
        scalers = [StandardScaler(), MinMaxScaler(), SplineTransformer(degree=4, n_knots=8),
                   RobustScaler(), PowerTransformer(method='yeo-johnson', standardize=True),
                   QuantileTransformer(), ]
        models_beyes = [GaussianNB(), BernoulliNB()]
        aSVR = [SVR(), NuSVR(), LinearSVR(), ]
        net_models = [ElasticNetCV(), ElasticNet(), ]
        models_regressor = [HuberRegressor(),
                            RandomForestRegressor(), DecisionTreeRegressor(), ExtraTreesRegressor(), SVR(),
                            LinearRegression(), AdaBoostRegressor(), SGDRegressor(), Lasso(), Ridge()]
        #CatBoostRegressor(), PoissonRegressor(),TweedieRegressor(), RANSACRegressor(),HistGradientBoostingRegressor()
        models_classifiers = [Perceptron(), RandomForestClassifier(),
                              DecisionTreeClassifier(), ExtraTreesClassifier(), LogisticRegression(),
                              LogisticRegressionCV(), SVC(),
                              LinearSVC(), RidgeClassifier(), IsolationForest(),
                              MLPClassifier(), AdaBoostClassifier(),
                              PassiveAggressiveClassifier()]  #CatBoostClassifier(),
        # NO MEMORY FOR
        experimental_models = [GaussianProcessClassifier()]  #[GradientBoostingClassifier(), ]
        last_estimation = 0
        if models_class == AI_MODELS.AI_REGRESSORS:
            models = models_regressor
        elif models_class == AI_MODELS.AI_BEYES:
            models = models_beyes
        #        elif models_class == AI_MODELS.AI_ML:
        #            models = models_ML
        elif models_class == AI_MODELS.AI_ALL:
            models = models_classifiers + models_regressor + net_models + models_beyes + experimental_models  #
        elif models_class == AI_MODELS.AI_CLASSIFIERS:
            models = models_classifiers
        elif models_class == AI_MODELS.AI_EXPERIMENTAL:
            models = experimental_models
        elif models_class == AI_MODELS.AI_ELASTIC:
            models = net_models
        for _ in range(len(models)):
            current_model = models[_]
            current_name = current_model.__repr__().replace('<', '').split(' ')[0].lower()
            current_name = current_name.split('.')[-1].replace('()', '')

            if current_name == 'elasticnetcv':
                search = HalvingGridSearchCV(current_model, HP_ELASTICNETCV,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='r2')
                class_classifiers = False
                current_model = search
            if current_name == 'sgdregressor':
                search = HalvingGridSearchCV(current_model, HP_SGD_REGESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            if current_name == 'tweedieregressor':
                search = HalvingGridSearchCV(current_model, HP_TWEEDIE_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            if current_name == 'ransacregressor':
                search = HalvingGridSearchCV(current_model, HP_RANSAC_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            if current_name == 'poissonregressor':
                search = HalvingGridSearchCV(current_model, HP_POISSON_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            if current_name == 'elasticnet':
                search = HalvingGridSearchCV(current_model, HP_ELASTICNET,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name in ['randomforestregressor', 'extratreesregressor']:
                search = HalvingGridSearchCV(current_model, HP_TREES_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'lasso':
                search = HalvingGridSearchCV(current_model, HP_LASSO,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'ridge':
                search = HalvingGridSearchCV(current_model, HP_RIDGE,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'histgradientboostingregressor':
                search = HalvingGridSearchCV(current_model, HP_HISTGRADBST_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name.__contains__('svr'):
                search = HalvingGridSearchCV(current_model, HP_SVR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            elif current_name.__contains__('nusvc'):
                search = HalvingGridSearchCV(current_model, HP_NUSVC,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('perceptron'):
                search = HalvingGridSearchCV(current_model, HP_PERCEPTRON,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search


            elif current_name.__contains__('linearsvc'):
                hyperset = HP_LINEAR_SVC_BINARY if not is_multiclass else HP_LINEAR_SVC_MULTI
                search = HalvingGridSearchCV(current_model, hyperset,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search
            elif current_name == 'svc':
                search = HalvingGridSearchCV(current_model, HP_SVC,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name == 'ridgeclassifier':
                search = HalvingGridSearchCV(current_model, HP_RIDGE_CLASSIFIER,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name == 'logisticregression':
                hyperset = HP_LOGISTIC_REGRESSION_MULTY if not is_multiclass else HP_LOGISTIC_REGRESSION_BINARY
                search = HalvingGridSearchCV(current_model, hyperset,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search
            elif current_name == 'decisiontreeregressor':
                search = HalvingGridSearchCV(current_model, HP_DECISIONTREE_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'xgbregressor':
                search = HalvingGridSearchCV(current_model, HP_XGBOOST_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'bernoullinb' or current_name == 'mutlinomialnb':
                search = HalvingGridSearchCV(current_model, HP_NAIVE_BAYES,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name == 'gaussiannb':
                search = HalvingGridSearchCV(current_model, HP_NAIVE_GAUSSIAN,
                                             verbose=1, n_jobs=-1, factor=2
                                             , cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            elif current_name in ['randomforestclassifier', 'decisiontreeclassifier', 'extratreesclassifier']:
                search = HalvingGridSearchCV(current_model, HP_FOREST_CLASSIFIERS,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('adaboostclassifier'):
                search = HalvingGridSearchCV(current_model, HP_ADABOOST_CLASSIFIER,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('adaboostregressor'):
                search = HalvingGridSearchCV(current_model, HP_ADABOOST_REGRESSOR,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='r2')
                current_model = search
                class_classifiers = False

            elif current_name.__contains__('mlpclassifier'):
                search = HalvingGridSearchCV(current_model, HP_MLP,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy')
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('gaussianprocessclassifier'):
                search = HalvingGridSearchCV(current_model, HP_GAUSSIAN_BOOST_CLASSIFIER,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy', aggressive_elimination=True)
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('gradientboostingclassifier'):
                search = HalvingGridSearchCV(current_model, HP_GRAD_CLASSIFIER,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy', aggressive_elimination=True)
                class_classifiers = True
                current_model = search

            elif current_name.__contains__('isolationforest'):
                search = HalvingGridSearchCV(current_model, HP_ISOLATION_FOREST,
                                             verbose=1, n_jobs=-1, factor=2,
                                             cv=cv_rsk, scoring='accuracy', aggressive_elimination=True)
                class_classifiers = True
                current_model = search

            elif current_name.__contains__("baggingregressor"):
                search = HalvingGridSearchCV(current_model, HP_BAGGING_REGRESSOR,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring='r2')
                current_model = search
                class_classifiers = False
            elif current_name.__contains__("linearregression"):
                search = HalvingGridSearchCV(current_model, HP_LINEAR_REGRESSION,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="r2")
                current_model = search
                class_classifiers = False
            elif current_name.__contains__("passiveaggressiveclassifier"):
                search = HalvingGridSearchCV(current_model, HP_PASSIVE_AGGRESSIVE_CLASSIFIER,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="accuracy")
                class_classifiers = True
                current_model = search

            elif current_name.__contains__("logisticregressioncv"):
                hyperset = HP_LOGISTICREGRESSIONCV_MULTI if is_multiclass else HP_LOGISTICREGRESSIONCV_BINARY
                search = HalvingGridSearchCV(current_model, hyperset,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="accuracy")
                class_classifiers = True
                current_model = search

            elif current_name.__contains__("catboostclassifier"):
                search = HalvingGridSearchCV(current_model, HP_CATBOOST_CLASSIFIER,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="accuracy")
                class_classifiers = True
                current_model = search
            elif current_name.__contains__("catboostregressor"):
                search = HalvingGridSearchCV(current_model, HP_CATBOOST_CLASSIFIER,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="r2")
                class_classifiers = False
                current_model = search
            elif current_name.__contains__("huberregressor"):
                search = HalvingGridSearchCV(current_model, HP_HUBER_REGRESSOR,
                                             verbose=1, n_jobs=-1,
                                             cv=cv_rsk, factor=2, scoring="r2")
                class_classifiers = False
                current_model = search

            for item in scalers:
                asyncio.run(write_log(message=f'Model {current_name}\n scaler:{item.__str__()} '
                                              f'\n started learning at:{dt.datetime.now()}', severity=SEVERITY.INFO))
                if current_name.__contains__("catboost") and item is None:
                    continue
                elif item is None:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                else:
                    X_train_scaled = item.fit_transform(X_train)
                    X_test_scaled = item.fit_transform(X_test)
                current_model.fit(X_train_scaled, Y_train)
                result = current_model.predict(X_test_scaled)
                if class_classifiers:
                    estimation_accuracy = metrics.accuracy_score(Y_test, result)
                    print(classification_report(Y_test, result))
                    cm = confusion_matrix(Y_test, result)
                    print(cm)
                else:
                    estimation_accuracy = metrics.r2_score(Y_test, result)
                estimation_accuracy = np.round(estimation_accuracy, 4)
                modelname = str(current_model.best_estimator_).split('(')[0]
                score = current_model.best_score_
                models_results.update(
                    {modelname: (score, estimation_accuracy)})
                asyncio.run(
                    write_log(message=f"Model:{modelname}\n scaler={item.__str__()}", severity=SEVERITY.INFO))
                asyncio.run(write_log(message=f"Good(>0.8):Medium(>0.6)"
                                              f" {estimation_accuracy}", severity=SEVERITY.INFO))
                if models_class == AI_MODELS.AI_CLASSIFIERS:
                    if best_model is None:
                        best_model = current_model
                        last_estimation = estimation_accuracy
                        best_scaler = item
                    elif estimation_accuracy > last_estimation:
                        best_model = current_model
                        last_estimation = estimation_accuracy
                        best_scaler = item
                else:
                    if best_model is None:
                        best_model = current_model
                        last_estimation = estimation_accuracy
                        best_scaler = item
                    elif estimation_accuracy > last_estimation:
                        best_model = current_model
                        last_estimation = estimation_accuracy
                        best_scaler = item
        gc.collect()
    asyncio.run(write_log(message=print(models_results), severity=SEVERITY.INFO))
    name = str(best_model.best_estimator_).split('(')[0]
    try:
        joblib.dump(best_model, f'{MODEL_STORE}{name}_aimodel.gzip', compress=3)
        with open(f"{MODEL_STORE}{name}_score.txt", "w+") as file:
            file.writelines(f"score:{last_estimation}")
        with open(f"{MODEL_STORE}{name}_model_parameters.json", "w+") as file:
            json.dump(best_model.best_params_, file)
        parameters = dict()
        parameters['scaler'] = best_scaler.__str__()
        parameters['labels'] = "None" if labels is None else labels
        parameters['boundaries'] = "None" if boundaries is None else boundaries
        parameters['multiclass'] = is_multiclass
        with open(f"{MODEL_STORE}{name}_preset_parameters.json", "w+") as file:
            json.dump(parameters, file)
    except Exception as ex:
        asyncio.run(write_log(message=f'{ex}  at:{dt.datetime.now()}', severity=SEVERITY.ERROR))
