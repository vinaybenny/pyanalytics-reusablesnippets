import os;
os.chdir("/home/vinay/Documents/code/ml_template/src");

import numpy as np;
from sklearn.cross_validation import StratifiedKFold;
from sklearn.cross_validation import KFold;
from pandas import read_csv, get_dummies;
from sklearn.preprocessing import LabelEncoder;


# Returns the k-fold split iterator, each of which can be 
# used to obtain different splits into training-validation.
def splitter(dataset, target, eval_size):    
    kf = StratifiedKFold(dataset[target], round(1./eval_size));    
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
    
    
if __name__ == "__main__":
    
    # Define dataset level variables here.
    dataset = read_csv("../data/train.csv");
    ycol="Survived";    
    validation_size=0.1;
    irrelevant_cols=list(["Name", "Ticket", "Cabin"]);
    
    # Data Cleaning
    # Drop columns that are not to be used in the model
    if len(irrelevant_cols) > 0:
        dataset = dataset.drop(irrelevant_cols, axis=1);
    
    # Drop NaNs from dataset or impute it.
    dataset = dataset.dropna();    
      
    # Split labels and covariates into different dataframes.r
    y = dataset.loc[:, ycol];
    
    # Convert all categorical covariates into one-hot encoding.
    X = binarizer(X);
    
    # Get first iteration of the kfold indices, use it for the train-validation split
    # Other iterations may be used later    
    kf = splitter(dataset, ycol, validation_size);
    train_indices, valid_indices = next(iter(kf));
    X_train, y_train = X.loc[train_indices], y.loc[train_indices];
        
    
    
    