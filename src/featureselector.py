from sklearn.decomposition import PCA;
from sklearn.decomposition import TruncatedSVD;
from sklearn.feature_selection import SelectKBest, f_classif, f_regression;
from sklearn.pipeline import FeatureUnion;
from pandas import DataFrame;


pcacomp = 4;    # Number of PCA components to pick
skbcomp = 129;    # Number of best original features to retain.

# Select which input dataset to use- binarized or labelled.
input_traindata = X_train;
input_validdata = X_valid;
input_testdata = X_test;

##################################################################################

print 'Applying PCA and automatic feature selection on the datasets...'    ;
# Define a pipeline of feature selection
pca = PCA(n_components=pcacomp);

# Set scoring function to classification if problem is classification, else to regression if continuous
if prob_type == 0 or prob_type == 1:
    skb = SelectKBest(score_func = f_classif, k=skbcomp);
    ycat = y;        
elif prob_type==2:
    skb = SelectKBest(score_func = f_regression, k=skbcomp);

combined_features = FeatureUnion([("skb", skb), ("pca", pca) ]);
combined_features.fit(input_traindata, y_train);

# Create the list of column names to be applied to transformed dataset
selected_features = list(input_traindata.columns[skb.get_support()]);
for i in range(1, pcacomp+1):
    selected_features.append("pca"+str(i) );

# Create the feature-selected datasets.
output_traindata = DataFrame(combined_features.transform(input_traindata));
output_traindata.columns = selected_features;
output_validdata = DataFrame(combined_features.transform(input_validdata));
output_validdata.columns = selected_features;
output_testdata = DataFrame(combined_features.transform(input_testdata));
output_testdata.columns = selected_features;
output_traindata.to_pickle("../output/train_pickle.pkl");
output_validdata.to_pickle("../output/valid_pickle.pkl");
output_testdata.to_pickle("../output/test_pickle.pkl");