import os;
os.chdir("/home/vinay/Documents/code/ml_template/src");

import numpy as np;
from sklearn.cross_validation import StratifiedKFold;
from sklearn.cross_validation import KFold;
from pandas import read_csv, get_dummies;
from sklearn.preprocessing import LabelEncoder;
import matplotlib.mlab as mlab;
import matplotlib.pyplot as plt;
plt.style.use('ggplot');


# Returns the k-fold split iterator, each of which can be 
# used to obtain different splits into training-validation.
def splitter(target, eval_size):    
    kf = StratifiedKFold(target, round(1./eval_size));    
    return kf;
   
# Convert categorical columns into one-hot encoding
def binarizer(dataframe):
    # Find all categorical columns from the dataframe
    all_cols = dataframe.columns;
    num_cols = dataframe._get_numeric_data().columns;
    cat_cols = list(set(all_cols) - set(num_cols));          
        
    for col in cat_cols:
        one_hot = get_dummies(dataframe[col], prefix=col);
        dataframe = dataframe.drop(col, axis=1);
        dataframe = dataframe.join(one_hot);

    return dataframe;    
    
# Convert categorical variables into numeric labels.
def labelizer(dataframe):
    # Find all categorical columns from the dataframe
    all_cols = dataframe.columns;
    num_cols = dataframe._get_numeric_data().columns;
    cat_cols = list(set(all_cols) - set(num_cols));    
    
    output = dataframe.copy();
        
    for col in cat_cols:
        lblenc = LabelEncoder(); 
        lblenc.fit(dataframe[col]);
        output[col] = lblenc.transform(dataframe[col]);
    
    return output;
    
# Plot histogram of the series and save as a ".png" file.
def plotdetailedhistogram(dataset, x, y):
    n, bins, patches = plt.hist(dataset[x], bins=50, normed=True, color="blue");        
    mean =  dataset[x].mean(axis=0);
    stddev = dataset[x].std(axis=0);
    normfit = mlab.normpdf(bins, mean, stddev);
    plt.plot(bins, normfit, "r--");
    plt.axvline(mean, color='black', linestyle='dashed', linewidth=2);
    plt.xlabel(x);
    plt.ylabel('Probability');
    plt.savefig("../output/plots/hist_" + x + ".png");        
    plt.close();
    
    
if __name__ == "__main__":
    
    # Define dataset level variables here.
    dataset = read_csv("../data/train.csv");
    ycol="Survived";    
    validation_size=0.1;
    irrelevant_cols=list(["Name"
                        , "Ticket"
                        , "Cabin"
                        ]);
    #########################################################################################
    # Data Cleaning and Transformations
    # Add new variables
    #<INSERT CODE HERE>    
                        
    # Drop columns with zero std deviation                    
    irrelevant_cols = irrelevant_cols + (dataset.std(axis=0, numeric_only=True) < 0.5)[(dataset.std(axis=0) == 0.0)].index.tolist();                    
                        
    # Drop columns that are not to be used in the model
    if len(irrelevant_cols) > 0:
        dataset = dataset.drop(irrelevant_cols, axis=1);
    
    # Raw dataset summaries
    #<INSERT CODE HERE> 
    
    
    # Treat missing values   
    dataset = dataset.dropna();
    
    # Post-treatment dataset summaries
    #<INSERT CODE HERE> 
    numeric_cols = list(dataset._get_numeric_data().columns);
    for col in numeric_cols:
        plotdetailedhistogram(dataset, col, ycol);
      
    # Split labels and covariates into different dataframes.
    y = dataset.loc[:, ycol];
    dataset = dataset.drop(ycol, axis=1);    
    
    # Convert all categorical covariates into one-hot encoding or labels.
    X = binarizer(dataset);
    #X = labelizer(dataset); 
    
    # Get first iteration of the k-fold indices, use it for the train-validation split
    # Other iterations may be used later    
    kf = splitter(y, validation_size);
    train_indices, valid_indices = next(iter(kf));
    X_train, y_train = X.loc[train_indices], y.loc[train_indices];
    X_valid, y_valid = X.loc[valid_indices], y.loc[valid_indices];
        
    
    
    