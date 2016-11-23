from sklearn.decomposition import PCA;
from sklearn.decomposition import TruncatedSVD;
from sklearn.feature_selection import SelectKBest, f_classif;
from sklearn.pipeline import FeatureUnion;
from pandas import DataFrame;


pcacomp = 100;    # Number of PCA components to pick
skbcomp = 5;    # Number of best original features to retain.

# Select which input dataset to use- binarized or labelled.
input_traindata = Xlab_train;
input_validdata = Xlab_valid;

##################################################################################

# Define a pipeline of feature selection
pca = PCA(n_components=pcacomp);
skb = SelectKBest(score_func = f_classif, k=skbcomp);
combined_features = FeatureUnion([("skb", skb), ("pca", pca) ]);
combined_features.fit(input_traindata, y_train);

# Create the list of column names to be applied to transformed dataset
selected_features = list(input_traindata.columns[skb.get_support()]);
for i in range(1, pcacomp+1):
    selected_features.append("pca"+str(i) );

# Create the feature-selected dataset.
output_traindata = combined_features.transform(input_traindata);
output_traindata = DataFrame(output_traindata);
output_traindata.columns = selected_features;

# Now apply the same transformation on the validation dataset.
output_validdata = DataFrame(combined_features.transform(input_validdata));
output_validdata.columns = selected_features;