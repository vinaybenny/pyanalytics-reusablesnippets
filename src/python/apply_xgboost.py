import xgboost as xgb;
from xgboost.sklearn import XGBRegressor, XGBClassifier;
from sklearn.model_selection import GridSearchCV;

# Select which input dataset to use
input_traindata = output_traindata;
input_validdata = output_validdata;
input_testdata = output_testdata;
param_grid = {'eta': [0.1],
            'gamma': [0.5],
            'max_depth': [3, 5],
            'min_child_weight': [ 7],
            'subsample': [1],
            'colsample_bytree': [1],
            'lambda': [1],
            'alpha': [0],
            'opt_rounds': [10],
            'early_stopping_rounds': [10]
            };


# Declare XGB learner
xgb_gsearch = GridSearchCV(estimator=XGBRegressor( objective='reg:linear', nthread=4, silent=False, seed=0 ),
                          scoring="neg_mean_absolute_error",
                          param_grid=param_grid,
                          n_jobs = 2,
                          cv=5,
                          verbose=2
);
xgb_gsearch.fit(input_traindata,y_train);