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
def binarizer(dataframe, cat_cols):
            
    for col in cat_cols:
        one_hot = get_dummies(dataframe[col], prefix=col);
        dataframe = dataframe.drop(col, axis=1);
        dataframe = dataframe.join(one_hot);

    return dataframe;    
    
# Convert categorical variables into numeric labels.
def labelizer(dataframe, cat_cols):
    
    output = dataframe.copy();
        
    for col in cat_cols:
        lblenc = LabelEncoder(); 
        lblenc.fit(dataframe[col]);
        output[col] = lblenc.transform(dataframe[col]);
    
    return output;
    
# Plot histogram of the series and save as a ".png" file.
def plotdetailedhistogram(dataset, x, y):
    num_bins = 50;
    fig, axes = plt.subplots(nrows=2, ncols=1);
    ax0, ax1 = axes.flat;  
    
    # Plot histogram  of the x column, with area under histogram normalized to 1.
    n, bins, patches = ax0.hist(dataset[x], bins=num_bins, normed=True, color="blue"); 
    mean =  dataset[x].mean(axis=0);
    stddev = dataset[x].std(axis=0);
    
    # Draw a normal curve fitting the 'x' variable. 
    normfit = mlab.normpdf(bins, mean, stddev);
    ax0.plot(bins, normfit, "r--", color="red");
    ax0.axvline(mean, color='black', linestyle='dashed', linewidth=2);
    ax0.set_xlabel(x);
    ax0.set_ylabel('Count');    
    colorlist=['green', 'black', 'brown', 'yellow', 'orange'];
    x_multi = [np.array(dataset.groupby(y)[x].get_group(i)) for i in [0, dataset.groupby(y)[x].ngroups-1]];
    n, bins, patches = ax1.hist(x_multi, num_bins,  normed=True, histtype='bar', alpha=0.6, color=colorlist[0:dataset.groupby(y)[x].ngroups]);
    
    # Loop through the data for each target class
    for i in range(0, dataset.groupby(y)[x].ngroups):
        normfit = mlab.normpdf(bins, dataset.groupby(y)[x].mean()[i], dataset.groupby(y)[x].std()[i] );
        ax1.plot(bins, normfit, "r--", color=colorlist[i]); 
        ax1.axvline(dataset.groupby(y)[x].mean()[i], color=colorlist[i], linestyle='dashed', linewidth=2);
    ax1.set_xlabel(x);
    ax1.set_ylabel('Count');
    plt.savefig("../output/plots/hist_" + x + ".png");        
    plt.close();
    

#############################################################################################    
if __name__ == "__main__":
    
    # Define dataset level variables here.
    full_dataset = read_csv("../data/train.csv");
    idcol = "PassengerId";
    ycol = "Survived";    
    validation_size=0.1;
    irrelevant_cols=list(["Name"
                        , "Ticket"
                        , "Cabin"
                        ]);
                        
    #########################################################################################
    # Data Cleaning and Transformations
    # Add new variables
    #<INSERT CODE HERE>    
    
    # Drop columns that are not to be used at all
    if len(irrelevant_cols) > 0:
        dataset = full_dataset.drop(irrelevant_cols, axis=1);                    
                        
    # Drop columns with zero std deviation- These add no information.                    
    irrelevant_cols = irrelevant_cols + (dataset.std(axis=0, numeric_only=True) < 0.5)[(dataset.std(axis=0) == 0.0)].index.tolist();                    
    
    # Classify attributes into numeric and categorical
    all_cols = dataset.columns;
    numeric_cols = list(dataset._get_numeric_data().columns);
    cat_cols = list(set(all_cols) - set(numeric_cols));  
    if idcol in numeric_cols:
        numeric_cols.remove(idcol);
    elif idcol in cat_cols:
        cat_cols.remove(idcol);    
                    
    # Raw dataset summaries.
    dataset_prestats = dataset.groupby(ycol).describe();    
    
    # Treat missing values   
    dataset = dataset.dropna();
    
    # Post-missing value treatment dataset summaries.
    dataset_poststats = dataset.groupby(ycol).describe();
    
    #########################################################################################       
    
    # Plot detailed histograms of variables    
    for col in numeric_cols:
        plotdetailedhistogram(dataset, col, ycol);
    
    
    # Split labels and covariates into different dataframes.
    y = dataset.loc[:, ycol];
    X = dataset.drop(ycol, axis=1);
     
    # Convert all categorical covariates into one-hot encoding or labels. Create both of later use.
    Xbin = binarizer(X, cat_cols);
    Xlab = labelizer(X, cat_cols);
    
    ##########################################################################################
    
    # Get first iteration of the k-fold indices, use it for the train-validation split
    # Other iterations may be used later    
    kf = splitter(y, validation_size);
    train_indices, valid_indices = next(iter(kf));
    Xbin_train, y_train = Xbin.loc[train_indices], y.loc[train_indices];
    Xbin_valid, y_valid = Xbin.loc[valid_indices], y.loc[valid_indices];
    Xlab_train, y_train = Xlab.loc[train_indices], y.loc[train_indices];
    Xlab_valid, y_valid = Xlab.loc[valid_indices], y.loc[valid_indices];
    