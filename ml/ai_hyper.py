import numpy as np

HP_ELASTICNET = {
    'alpha': np.round(np.arange(0.01, 0.1, 0.05), 2),
    'l1_ratio': np.round(np.arange(0.0, 1.0, 0.1), 2),
    'fit_intercept': (True, False),
    'max_iter': range(10, 1000, 10),
    'positive': (True, False),
    'selection': ('cyclic', 'random')
}

HP_ELASTICNETCV = {
    'l1_ratio': np.round(np.arange(0.1, 1.0, 0.1), 2),
    'fit_intercept': (True, False),
    'positive': (True, False),
    'selection': ('cyclic', 'random'),
    'cv': range(1, 10),
    'n_alphas': range(1, 200, 5),
    'max_iter': range(1000, 2000, 100)
}

HP_TREES_REGRESSOR = {
    'n_estimators': range(1, 10),
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': range(1, 15),
    'max_depth': range(3, 40)
}
HP_LASSO = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
}
HP_RIDGE = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
}
HP_HISTGRADBST_REGRESSOR = {
    'l2_regularization': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.79, 1],
    'max_iter': [200]
}
HP_SVC = {
    'C': [1, 10, 100],
    'kernel': ['poly', 'linear', 'rbf']
}

HP_LOGISTIC_REGRESSION_BINARY = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'solver': ['saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'],
    #    'multi_class': ['multinomial','ovr'],
    'C': [0.001, 0.01, 0.05, 0.07, 0.1, 0.5, 0.7, 0.75, 0.9, 1.0],
    'tol': (0.000001, 0.0001, 0.001, 0.01, 0.1, 0.9),
    'fit_intercept': (True, False),
    'max_iter': range(100, 1000, 10)

}

HP_LOGISTIC_REGRESSION_MULTY = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'solver': ['saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'],
    #    'multi_class': ['multinomial','ovr'],
    'C': [0.001, 0.01, 0.05, 0.07, 0.1, 0.5, 0.7, 0.75, 0.9, 1.0],
    'tol': (0.000001, 0.0001, 0.001, 0.01, 0.1, 0.9),
    'fit_intercept': (True, False),
    'max_iter': range(100, 1000, 10),
    "multi_class": ["multinomial"]

}

HP_DECISIONTREE_REGRESSOR = {
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': range(3, 20),
    'max_depth': range(3, 20),
    'max_leaf_nodes': range(3, 20),

}

HP_XGBOOST_REGRESSOR = {
    "max_depth": [3, 4, 5, 7],
    "gamma": [0, 0.25, 1],
    "scale_pos_weight": [1, 3, 5]
}
HP_NAIVE_BAYES = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
    'fit_prior': [True, False]
}
HP_NAIVE_GAUSSIAN = {
    'var_smoothing': [0.001, 0.01, 0.1, 1, 2, 5],
}

HP_SVR = {
    # 7 specified parameters
    'C': [0.001, 0.01, 0.1, 1],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['poly', 'linear', 'rbf'],
    'epsilon': [0.01, 0.1, 0.5, 0.9]
}

HP_FOREST_CLASSIFIERS = {
    'criterion': ('gini', 'log_loss', 'entropy'),
    'min_samples_split': range(2, 10),
    'max_depth': range(2, 40),
    'min_samples_leaf': range(2, 40),
}
HP_ADABOOST_CLASSIFIER = {
    'n_estimators': range(2, 50),
    'algorithm': ('SAMME', 'SAMME.R'),
}

HP_ADABOOST_REGRESSOR = {
    'n_estimators': range(2, 50),
    'loss': ('linear', 'square', 'exponential'),
}

HP_MLP = {
    "activation": ('identity', 'logistic', 'tanh', 'relu'),
    "solver": ('lbfgs', 'sgd', 'adam'),
    "max_iter": range(10, 100)
}
HP_GAUSSIAN_BOOST_CLASSIFIER = {
    "n_restarts_optimizer": range(0, 10),
    "max_iter_predict": range(0, 10),
    "multi_class": ('one_vs_rest', 'one_vs_one')
}
HP_GRAD_CLASSIFIER = {
    "loss": ("log_loss", "exponential"),
    "n_estimators": range(100, 200, 2),
    "criterion": ("fridman_mse", "squared_error"),
    "min_samples_split": range(2, 50),
    'min_samples_leaf': range(2, 50),
    'max_depth': range(2, 10),
    'ccp_alpha': np.arange(0.0, 1.0, 0.1)
}

HP_SGD_REGESSOR = {
    'loss': ('squared_error', 'huber', 'epslion_insensitive', 'squared_epsilon_insensitive'),
    'penalty': ('l2', 'l1', 'elasticnet'),
    'max_iter': range(1000, 2000, 5),
    'learning_rate': ('optimal', 'constant', 'invscaling', 'adaptive'),
    "tol": (0.0001, 0.001, 0.005, 0.01, 0.1),
    "epsilon": (0.001, 0.01, 0.05, 0.1, 0.5, 0.9)
}

HP_BAGGING_REGRESSOR = {
    "n_estimators": range(1, 20)
}

HP_LINEAR_REGRESSION = {
    "fit_intercept": (True, False),
}

HP_PASSIVE_AGGRESSIVE_CLASSIFIER = {
    "fit_intercept": (True, False),
    "max_iter": range(1000, 2000, 5),
}

HP_LOGISTICREGRESSIONCV_BINARY = {
    "Cs": range(1, 100, 5),
    "fit_intercept": (True, False),
    "penalty": ("l1", "l2", "elasticnet"),
    "solver": ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
    "refit": (True, False),
}
HP_LOGISTICREGRESSIONCV_MULTI = {
    "Cs": range(1, 100, 5),
    "fit_intercept": (True, False),
    "penalty": ("l1", "l2", "elasticnet"),
    "solver": ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
    "refit": (True, False),
    "multi_class": ["multinomial"]
}

HP_CATBOOST_CLASSIFIER = {
    "iterations": range(10, 100, 10),
    "learning_rate": np.round(np.arange(0.10, 0.90, 0.05), 2)
}
HP_CATBOOST_REGRESSOR = {
    "n_estimators": range(10, 100, 10),
    "learning_rate": np.round(np.arange(0.10, 0.90, 0.10), 2),
    "depth": range(1, 10)
}

HP_NUSVC = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    "degree": range(3, 10),
    "gamma": ('scale', 'auto'),
    "decision_function": ('ovo', 'ovr'),
    "nu": [0.99, 0.999, 0.9999, 1.0]
}
HP_LINEAR_SVC_BINARY = {
    "penalty": ("l1", "l2"),
    "loss": ("hinge", "squared_hinge"),
    "C": np.round(np.arange(0.1, 5.0, 0.1)),
    "multi_class": ["ovr"],
    "fit_intercept": (True, False),
    "max_iter": range(1000, 2000, 100)
}
HP_LINEAR_SVC_MULTI = {
    "penalty": ("l1", "l2"),
    "loss": ("hinge", "squared_hinge"),
    "C": np.round(np.arange(0.1, 5.0, 0.1)),
    "multi_class": ["crammer_singer"],
    "fit_intercept": (True, False),
    "max_iter": range(1000, 2000, 100)
}

HP_RIDGE_CLASSIFIER = {
    "alpha": np.arange(1.0, 10.0, 0.2),
    "fit_intercept": (True, False),
    "max_iter": range(10, 1000, 50),
    "solver": ('svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'),
    "tol": (0.0001, 0.001, 0.005, 0.01, 0.1)
}

HP_HUBER_REGRESSOR = {
    "epsilon": np.arange(1.0, 3.0, 0.15),
    "max_iter": range(100, 1000, 25),
    "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 0.9, 1],
    "fit_intercept": (True, False)
}
HP_ISOLATION_FOREST = {
    "n_estimators": range(10, 100, 5),
    "bootstrap": (True, False),
}
HP_TWEEDIE_REGRESSOR = {
    "power": [0, 1, 1.2, 2, 3],
    "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 0.9, 1],
    "link": ("identity", "log"),
    "solver": ('lbgfs', 'newton-cholesky'),
    "tol": (0.0001, 0.001, 0.005, 0.01, 0.1)
}
HP_POISSON_REGRESSOR = {
    "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 0.9, 1],
    "fit_intercept": (True, False),
    "solver": ('lbfgs', 'newton-cholesky'),
}
HP_RANSAC_REGRESSOR = {
    "min_samples": range(0, 10),
    "max_trials": range(0, 10),
    "stop_probability": np.arange(0.1, 0.99, 0.1),
    "loss": ('absolute_error', 'squared_error')
}
HP_THEIL_SEN_REGRESSOR = {
    "fit_intercept": (True, False),
    "tol": (0.0001, 0.001, 0.005, 0.01, 0.1),
}

HP_PERCEPTRON = {
    "penalty": ('l2', 'l1', 'elasticnet'),
    "alpha": [0, 0.0001, 0.001, 0.01, 0.1, 0.9, 1],
    "fit_intercept": (True, False),
    "shuffle": (True, False),
    "class_weight":["balanced"]
}
