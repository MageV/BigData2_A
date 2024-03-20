import numpy as np

param_elastic = {
    'alpha': np.round(np.arange(0.01, 0.1, 0.05), 2),
    'l1_ratio': np.round(np.arange(0.0, 1.0, 0.1), 2),
    'fit_intercept': (True, False),
    'max_iter': range(10, 1000, 10),
    'positive': (True, False),
    'selection': ('cyclic', 'random')
}

param_elastic_cv = {
    'l1_ratio': np.round(np.arange(0.1, 1.0, 0.1), 2),
    'fit_intercept': (True, False),
    'positive': (True, False),
    'selection': ('cyclic', 'random'),
    'cv': range(1, 10),
    'n_alphas': range(1, 200, 5),
    'max_iter':range(1000,2000,100)
}

param_rf = {
    'n_estimators': range(1, 10),
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': range(1, 15),
    'max_depth': range(2, 20)
}
param_lasso = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
}
param_ridge = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
}
param_hbr = {
    'l2_regularization': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.79, 1],
    'max_iter': [200]
}
param_svc = {
    'C': [1, 10, 100],
    'kernel': ['poly', 'linear', 'rbf']
}

param_lr = {
    'penalty': ['l2', 'elasticnet'],
    'solver': ['saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'],
    #    'multi_class': ['multinomial','ovr'],
    'C': [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0],
}

param_dtr = {
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': range(3, 20),
    'max_depth': range(3, 20),
    'max_leaf_nodes': range(3, 20),

}

param_xgboost = {
    "max_depth": [3, 4, 5, 7],
    "gamma": [0, 0.25, 1],
    "scale_pos_weight": [1, 3, 5]
}
param_gaussian_cat_multi = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
    'fit_prior': [True, False]
}
param_gaussian_nb = {
    'var_smoothing': [0.001, 0.01, 0.1, 1, 2, 5],
}

param_svr = {
    # 7 specified parameters
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['poly', 'linear', 'rbf']
}

param_rfc = {
    'criterion': ('gini', 'log_loss', 'entropy'),
    'min_samples_split': range(2, 10),
    'max_depth': range(2, 10),
    'min_samples_leaf': range(2, 10),
}
param_ada_classifier = {
    'n_estimators': range(2, 50),
    'algorithm': ('SAMME', 'SAMME.R'),
}

param_ada_regressor = {
    'n_estimators': range(2, 50),
    'loss': ('linear', 'square', 'exponential'),
}

param_mlp = {
    "activation": ('identity', 'logistic', 'tanh', 'relu'),
    "solver": ('lbfgs', 'sgd', 'adam'),
    "max_iter": range(10, 100)
}
param_gauss_proc = {
    "n_restarts_optimizer": range(0, 10),
    "max_iter_predict": range(0, 10),
    "multi_class": ('one_vs_rest', 'one_vs_one')
}
param_grad_boost = {
    "loss": ("log_loss", "exponential"),
    "n_estimators": range(100, 200, 2),
    "criterion": ("fridman_mse", "squared_error"),
    "min_samples_split": range(2, 50),
    'min_samples_leaf': range(2, 50),
    'max_depth': range(2, 10),
    'ccp_alpha': np.arange(0.0, 1.0, 0.1)
}

param_sgd_regr = {
    'loss': ('squared_error', 'huber', 'epslion_insensitive', 'squared_epsilon_insensitive'),
    'penalty': ('l2', 'l1', 'elasticnet'),
    'max_iter': range(1000, 2000, 5),
    'learning_rate': ('optimal', 'constant', 'invscaling')
}

param_bagging_regr = {
    "n_estimators": range(1, 20)
}

param_linear_regr = {
    "fit_intercept": (True, False),
}

param_pass_agg_clf = {
    "fit_intercept": (True, False),
    "max_iter": range(1000, 2000, 5),
}

param_logr_cv = {
    "Cs": range(1, 20),
    "fit_intercept": (True, False),
    "penalty": ("l1", "l2", "elasticnet"),
    "solver": ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
    "refit": (True, False),
}

param_cat_bst = {
    "iterations": range(10, 100, 10),
    "learning_rate": np.round(np.arange(0.10, 0.90, 0.05), 2)
}
param_cat_bst_rgr = {
    "n_estimators": range(10, 100, 10),
    "learning_rate": np.round(np.arange(0.10, 0.90, 0.10), 2),
    "depth": range(1, 10)
}

param_nu_svc={
    "kernel":['linear','poly','rbf','sigmoid','precomputed'],
    "degree":range(3,10),
    "gamma":('scale','auto'),
    "decision_function":('ovo','ovr'),
    "nu":[0.99,0.999,0.9999,1.0]
}
param_linear_svc={
    "penalty":("l1","l2"),
    "loss":("hinge","squared_hinge"),
    "C":np.round(np.arange(0.1,5.0,0.1)),
    "multi_class":("ovr","crammer_singer"),
    "fit_intercept":(True,False),
    "max_iter":range(1000,2000,100)
}