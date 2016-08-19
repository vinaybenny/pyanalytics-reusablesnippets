from sklearn.decomposition import PCA;
from sklearn.decomposition import TruncatedSVD;
from sklearn.feature_selection import SelectKBest, f_classif;
from sklearn.pipeline import FeatureUnion;
from pandas import DataFrame;


pcacomp = 5;    # Number of PCA components to pick
skbcomp = 5;    # Number of best original features to retain.

# Select which input dataset to use- binarized or labelled.
inputdata = Xlab_train;

##################################################################################

# Define a pipeline of feature selection
pca = PCA(n_components=pcacomp);
skb = SelectKBest(score_func = f_classif, k=skbcomp);
combined_features = FeatureUnion([("skb", skb), ("pca", pca) ]);
combined_features.fit(inputdata, y_train);

# Create the list of column names to be applied to transformed dataset
selected_features = list(inputdata.columns[skb.get_support()]);
for i in range(1, pcacomp+1):
    selected_features.append("pca"+str(i) );

# Create the feature-selected dataset.
outputdata = combined_features.transform(inputdata);
outputdata = DataFrame(outputdata);
outputdata.columns = selected_features;