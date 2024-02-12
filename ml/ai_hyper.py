param_ridge = {
    'solver': ['svd', 'cholesky', 'lsqr', 'sag'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'fit_intercept': [True, False],
}
param_lasso = {
    'random_state': range(1, 21),
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'fit_intercept': [True, False],
    'selection': ['cyclic', 'random'],
}
param_elastic={
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    'l1-ratio': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
    'fit_intercept': [True, False],
    'selection': ['cyclic', 'random'],
}
param_lr = {
        'penalty': ('l1', 'l2', 'elasticnet'),
        'solver': ('saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'),
        'class_weight':['balanced'],
        'random_state':range(0,5),
        'C': (0.1, 0.15,0.25,0.35, 0.45, 0.55,0.65,0.75, 0.9,1.0,1.5,2.0),
        'multi_class':['multinomial','ovr']
    }
