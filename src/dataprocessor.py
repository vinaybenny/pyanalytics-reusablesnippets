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
import matplotlib.cm as cm;
plt.style.use('ggplot');
from pylab import rcParams;
rcParams['figure.figsize'] = 20, 10


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
    
    # Now plot the histogram of column x specific to each level in target column y.
    # First, get the data grouped according to 
    grouped_data=dataset.groupby(y)[x];
    colorlist=cm.Spectral([float(var) / grouped_data.ngroups for var in range(grouped_data.ngroups)])
    
    x_multi=[];
    x_multi_name=[];
    for name, group in grouped_data:
        x_multi.append( np.array(grouped_data.get_group(name)) ); 
        x_multi_name.append(name);
    n, bins, patches = ax1.hist(x_multi, num_bins,  normed=True, histtype='bar', alpha=1.0, 
                                label=x_multi_name, color=colorlist[0:grouped_data.ngroups]);
    
    # Now draw a distribution curve for x column for each level of column y.
    # Loop through the data for each target class
    for i in range(0, grouped_data.ngroups):
        normfit = mlab.normpdf(bins, grouped_data.mean()[i], grouped_data.std()[i] );
        ax1.plot(bins, normfit, "r--", color=colorlist[i]); 
        ax1.axvline(grouped_data.mean()[i], color=colorlist[i], linestyle='dashed', linewidth=2);
        
    ax1.set_xlabel(x);
    ax1.set_ylabel('Count');
    ax1.legend();
    plt.savefig("../output/plots/hist_" + x + ".png", dpi=500);        
    plt.close();
    

#############################################################################################    
if __name__ == "__main__":
    
    print 'Starting execution...';
    print 'Reading training data...'    
    # Define dataset level variables here.
    full_dataset = read_csv("../data/train.csv");
    idcol = "id";
    ycol = "loss";    
    validation_size=0.2;
    irrelevant_cols=list([]);
    prob_type = 1; # 0 for binary classification, 1 for regression
    quantiles=4; # variable for target column quantiles in case target variable is continuous(for grouped desriptive statistics creation)
    print 'Read completed.';
                        
    #########################################################################################  
    print 'Dropping irrelevant and useless columns...';
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
    print 'Collecting pre-transformation variable stats...'    ;
    dataset_prestats, dataset_grouped_prestats = descstats(dataset, ycolcat);
    
    # Data Cleaning and Transformations
    # Add new variables
    #<INSERT CODE HERE>    
    # Treat missing values   
    dataset = dataset.dropna();
    
    # Post-missing value treatment dataset summaries.
    print 'Collecting post-transformation variable stats...' ;
    dataset_poststats, dataset_grouped_poststats = descstats(dataset, ycolcat);
    
      
    
    # Plot detailed histograms of variables    
    print 'Creating Histograms for numeric variables...';
    for col in numeric_cols:
        if not(col == ycol):
            plotdetailedhistogram(dataset, col, ycolcat );
    
    #########################################################################################     
    # Split labels and covariates into different dataframes.
    print 'Splitting covariates and target...';
    y = dataset.loc[:, ycol];
    y_cat = dataset.loc[:, ycolcat];
    X = dataset.drop([ycol,ycolcat], axis=1);
     
    # Convert all categorical covariates into one-hot encoding or labels. Create both for later use.
    print 'Saving One-Hot encoded and labelised datasets...';
    Xbin = binarizer(X, cat_cols);
    Xlab = labelizer(X, cat_cols);
    
    ##########################################################################################
    
    # Get first iteration of the k-fold indices, use it for the train-validation split
    # Other iterations may be used later  
    print 'Splitting training data into training and validation sets...';
    skf = StratifiedKFold(n_splits=int(1./validation_size), shuffle=True);
    skf.get_n_splits(X, y);
    
    train_indices, valid_indices = next(iter(skf.split(X, y_cat)));
    Xbin_train, y_train = Xbin.iloc[train_indices], y.iloc[train_indices];
    Xbin_valid, y_valid = Xbin.iloc[valid_indices], y.iloc[valid_indices];
    Xlab_train, y_train = Xlab.iloc[train_indices], y.iloc[train_indices];
    Xlab_valid, y_valid = Xlab.iloc[valid_indices], y.iloc[valid_indices];
    
    print 'Completed data processing.';
    
    