import xgboost;
import itertools;
from sklearn.metrics import mean_absolute_error;

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds)-1, np.exp(labels))-1


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


# Select which input dataset to use
input_traindata = output_traindata;
input_validdata = output_validdata;
input_testdata = output_testdata;
param_grid = expand_grid(
    {'eta': [0.05,0.07, 0.1],
    'gamma': [1],
    'max_depth': [5, 7, 10],
    'min_child_weight': [3, 5, 7],
    'subsample': [1],
    'colsample_bytree': [1],
    'lambda': [1],
    'alpha': [0],
    'opt_rounds': [1500],
    'early_stopping_rounds': [10],
    # Initialise model evaluation measures
    'train_measure': [0.0],
    'valid_measure': [0.0]
    });


###############################################################################
dtrain = xgboost.DMatrix(data=input_traindata, label=y_train);
dval = xgboost.DMatrix(data = output_validdata, label=y_valid);  
dtest = xgboost.DMatrix(data = output_testdata); 
    
for index, row in param_grid.iterrows():
    print 'Model number: '+ str(index);
    params = {
          "booster": "gbtree",
          "silent": 1,
          "eta": row['eta'],
          "max_depth": int(row['max_depth']),
          "subsample": row['subsample'],
          "colsample_bytree": row['colsample_bytree'],
          "gamma": row['gamma'],
          "min_child_weight": row['min_child_weight'],
          "lambda": row['lambda'],
          "alpha": row['alpha']
          };   
    res = xgboost.cv(params=params,
                     dtrain = dtrain, 
                     feval=evalerror,
                     num_boost_round=int(row['opt_rounds']), 
                     nfold=5,
                     metrics={'mae'}, 
                     early_stopping_rounds = int(row['early_stopping_rounds']),
                     verbose_eval=True,
                     seed = 0);
                     
    if res.shape[0]==int(row['opt_rounds']):
        param_grid.set_value(index, 'opt_rounds', int(row['opt_rounds']) );
        param_grid.set_value(index,'valid_measure', res.iloc[res.shape[0]-1 , 0 ]);
        param_grid.set_value(index,'train_measure', res.iloc[res.shape[0]-1 , 2 ]);
    else:
        param_grid.set_value(index,'opt_rounds', (res.shape[0]-int(row['early_stopping_rounds']) ) / (1- (1/5) ));
        param_grid.set_value(index,'valid_measure', res.iloc[res.shape[0]-int(row['early_stopping_rounds']) , 0 ]);
        param_grid.set_value(index,'train_measure', res.iloc[res.shape[0]-int(row['early_stopping_rounds']) , 2 ]);

param_grid.to_csv("../output/xgboost_cv_output.csv");
best_model_idx = param_grid["train_measure"].idxmin(axis=1);
best_params = {
          "booster": "gbtree",
          "silent": 1,
          "eta": param_grid['eta'][best_model_idx],
          "max_depth": int(param_grid['max_depth'][best_model_idx]),
          "subsample": param_grid['subsample'][best_model_idx],
          "colsample_bytree": param_grid['colsample_bytree'][best_model_idx],
          "gamma": param_grid['gamma'][best_model_idx],
          "min_child_weight": param_grid['min_child_weight'][best_model_idx],
          "lambda": param_grid['lambda'][best_model_idx],
          "alpha": param_grid['alpha'][best_model_idx]
          };

xgb_model=xgboost.train(params = best_params,
              dtrain=dtrain, 
              feval= evalerror,
              num_boost_round=param_grid['opt_rounds'][best_model_idx], 
              evals=[(dval, "valid")], 
              early_stopping_rounds=int(row['early_stopping_rounds']), 
              verbose_eval=True);
test_pred = pd.Series(data = xgb_model.predict(dtest), name=ycol)

    
xgboost_output = pd.concat([pd.Series(covariates.loc[covariates['rowtype'] == "TEST"][idcol].reset_index()[idcol], name = idcol), np.exp(test_pred)-1 ], 
           axis=1);
xgboost_output.to_csv("../output/xgboost_pred_output.csv", index=False);
