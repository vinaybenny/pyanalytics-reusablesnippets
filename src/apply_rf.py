from sklearn.ensemble import RandomForestRegressor;
from sklearn.model_selection import GridSearchCV;

###############################################################################
input_traindata = pd.read_pickle("../output/train_pickle.pkl");
input_validdata = pd.read_pickle("../output/valid_pickle.pkl");
input_testdata = pd.read_pickle("../output/test_pickle.pkl");

###############################################################################

# Attempt to fit a random forest model on the training dataset
rf_model = RandomForestRegressor(criterion="mae", verbose=1, n_jobs=-1);
clf = GridSearchCV(rf_model,
                   {'max_depth': [2],
                    'n_estimators': [120,300,500,800,1200],
                    'max_features':[None, 5, 15, 20, 30],
                    'min_samples_leaf':[1,2,5,10],
                    'min_samples_split':[2,5,10],
                    'max_features':["sqrt", "log2", None]
                    }, verbose=2)

clf.fit(input_traindata, y_train);
X_selected = clf.transform(output_testdata);

X_selected.to_csv('../data/out.csv');