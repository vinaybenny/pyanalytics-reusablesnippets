import os;
os.chdir("C:\\Users\\vinay.benny\\Documents\\Kaggle\\Allstate\\src");

import numpy as np;
import pandas as pd;
from sklearn.model_selection import StratifiedKFold;
from sklearn.model_selection import KFold;
from pandas import read_csv, get_dummies;
from sklearn.preprocessing import LabelEncoder;
import matplotlib.mlab as mlab;
import matplotlib.pyplot as plt;
plt.style.use('ggplot');
from pylab import rcParams;
rcParams['figure.figsize'] = 20, 10


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
    
def descstats(dataset, ycol_cat):
    dataset_prestats = dataset.describe(); # Overall statistics
    dataset_grouped_prestats = dataset.groupby(ycol_cat).describe(); # statistics grouped on target variable categories    
    return dataset_prestats, dataset_grouped_prestats;
    
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
    
    x_multi=[];
    for name, group in dataset.groupby(y)[x]:
        x_multi.append( np.array(dataset.groupby(y)[x].get_group(name)) );
        
    #x_multi = [np.array(dataset.groupby(y)[x].get_group(i)) for i in [0, dataset.groupby(y)[x].ngroups-1]];
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
    idcol = "Id";
    ycol = "loss";    
    validation_size=0.2;
    irrelevant_cols=list([]);
    prob_type = 1; # 0 for binary classification, 1 for regression
    quantiles=4; # variable for target column quantiles in case target variable is continuous(for grouped desriptive statistics creation)
    
                        
    #########################################################################################  
    # Find and add columns with zero std deviation to irrelevant columns- These add no information.                    
    irrelevant_cols = irrelevant_cols + (full_dataset.std(axis=0, numeric_only=True) < 0.5)[(full_dataset.std(axis=0) == 0.0)].index.tolist();
    
    # Drop columns that are not to be used at all
    if len(irrelevant_cols) > 0:
        dataset = full_dataset.drop(irrelevant_cols, axis=1);
    else:
        dataset=full_dataset;

    # Classify remaining attributes into numeric and categorical
    all_cols = dataset.columns;
    numeric_cols = list(dataset._get_numeric_data().columns);
    cat_cols = list(set(all_cols) - set(numeric_cols));  
    if idcol in numeric_cols:
        numeric_cols.remove(idcol);
    elif idcol in cat_cols:
        cat_cols.remove(idcol);                                             

    # Set outcome variable to categorical if problem is classification, else to float64
    if prob_type == 0:
        dataset[ycol] = dataset[ycol].astype("category");
        dataset[ycol + "_cat"] = dataset[ycol];
    elif prob_type==1:
        dataset[ycol] = dataset[ycol].astype("float64");  
        dataset[ycol + "_cat"] = pd.qcut(dataset[ycol], quantiles); # Quantile based cuts for target column   
        ycolcat = ycol + "_cat" # Column Name for categorical representation of target column
    
    # Descriptive Stats for numerical variables in pre-transformation dataset    
    dataset_prestats, dataset_grouped_prestats = descstats(dataset, ycolcat);
    
    # Data Cleaning and Transformations
    # Add new variables
    #<INSERT CODE HERE>    
    # Treat missing values   
    dataset = dataset.dropna();
    
    # Post-missing value treatment dataset summaries.
    dataset_poststats, dataset_grouped_poststats = descstats(dataset, ycolcat);
    
      
    
    # Plot detailed histograms of variables    
    for col in numeric_cols:
        plotdetailedhistogram(dataset, col, ycolcat );
    
    #########################################################################################     
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
    Xbin_train, y_train = Xbin.iloc[train_indices], y.iloc[train_indices];
    Xbin_valid, y_valid = Xbin.iloc[valid_indices], y.iloc[valid_indices];
    Xlab_train, y_train = Xlab.iloc[train_indices], y.iloc[train_indices];
    Xlab_valid, y_valid = Xlab.iloc[valid_indices], y.iloc[valid_indices];
    
    
    