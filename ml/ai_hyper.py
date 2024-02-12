param_elastic = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 0.75, 0.79, 1, 1.5, 2, 5],
    'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    'kernel': ('linear', 'rbf'),
    'C': [1, 10, 100]
}
param_lr = {
    'penalty': ('l1', 'l2', 'elasticnet'),
    'solver': ('saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'),
    'multi_class': ('ovr', 'multinomial'),
    'class_weight': ['balanced'],
    'random_state': range(0, 10),
    'C': (0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0, 1.5, 2.0),
}

param_dtr = {
    'max_features': ['log2', 'sqrt'],
    'min_samples_split': range(1, 20),
    'max_depth': range(2, 20),
    'max_leaf_nodes': range(2, 20)
}
