import os;
import numpy as np;
import pandas as pd;
from collections import defaultdict;
from sklearn.model_selection import StratifiedKFold;
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
def labelizer(dataframe, ordinal_cols):
    #Create a dictionary of label encoder object for each column in the input dataset
    lblenc_dict = defaultdict(LabelEncoder);
    
    # Encoding each categorical variable
    dataframe[ordinal_cols].apply(lambda x: lblenc_dict[x.name].fit(x)); 
    return lblenc_dict;
    
def descstats(dataset, ycol_cat=None):
    if ycol_cat == None or len(ycol_cat)==0:    
        dataset_prestats = dataset.describe(); # Overall statistics
    else:
        dataset_prestats = dataset.groupby(ycol_cat).describe(); # statistics grouped on target variable categories    
    return dataset_prestats;
    
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
    os.chdir("C:\\Users\\vinay.benny\\Documents\\Kaggle\\Allstate\\src");
    print 'Starting execution...';
    print 'Reading training and test data...'    
    
    # Define data processor level variables here.
    full_dataset = read_csv("../data/train.csv");
    test_dataset = read_csv("../data/test.csv");
    idcol = "id";
    ycol = "loss";    
    validation_size=0.2;
    irrelevant_cols=list([]);
    # if there are ordinal columns among the covariates, define these here. These will be converted to numeric labels.
    ordinal_cols=list([]);
    # 0 for binary classification, 1 for multinomial classification, 2 for regression
    prob_type = 2; 
    # variable for target column quantiles in case target variable is continuous(for grouped desriptive statistics creation)
    quantiles=4; 
    print 'Read completed.';
                        
    #########################################################################################     
    # Split train dataset labels and covariates into different dataframes.
    print 'Splitting covariates and target...';
    y = full_dataset.loc[:, ycol];    
    X = full_dataset.drop([ycol], axis=1);
    
    # Set outcome variable to categorical if problem is classification, else to float64
    if prob_type == 0 or prob_type == 1:
        y = y.astype("category");
        ycat = y;        
    elif prob_type==2:
        y = y.astype("float64");  
        ycat = pd.qcut(y, quantiles); # Quantile based cuts for target column  
        
    ##########################################################################################    
    # Get first iteration of the k-fold indices, use it for the train-validation split
    # Other iterations may be used later  
    print 'Splitting training data into training and validation sets...';
    skf = StratifiedKFold(n_splits=int(1./validation_size), shuffle=True);
    skf.get_n_splits(X, y);
    
    train_indices, valid_indices = next(iter(skf.split(X, ycat)));
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices];
    X_valid, y_valid = X.iloc[valid_indices], y.iloc[valid_indices];
    X_test = test_dataset;
    
    #########################################################################################  
    # DATA CLEANING AND TRANSFORMATION, DESCRIPTIVE STATS
    print 'Applying cleaning rules and data transformations...';    
    # Treat missing values   
    X_train = X_train.dropna();
    X_valid = X_valid.dropna();
    
    # Combine train, valid and test covariates to create a consolidated covariate set
    X_train=X_train.join(pd.Series('TRAIN', index=X_train.index, name = 'rowtype'));
    X_valid=X_valid.join(pd.Series('VALID', index=X_valid.index, name = 'rowtype'));
    X_test=X_test.join(pd.Series('TEST', index=X_test.index, name = 'rowtype'));  
    
    covariates = pd.concat([X_train, X_valid, X_test], axis=0, ignore_index=True); 
    # If id column does not exist, create one.    
    if (idcol is None) or ( len(idcol) == 0 ):
        idcol = 'id';
        covariates=covariates.join(pd.Series( range(1, len(covariates) + 1,1), index=covariates.index, name = idcol ));
    
    # Find and add columns with zero std deviation to irrelevant columns- These add no information.                    
    irrelevant_cols = irrelevant_cols + (covariates.std(axis=0, numeric_only=True) < 0.5)[(covariates.std(axis=0) == 0.0)].index.tolist();
    
    # Data Cleaning and Transformations
    # Add new variables
    #<INSERT CODE HERE>
       
    # Classify attributes into numeric, categorical and ordinal
    all_cols = list( set(covariates.columns) - set(['rowtype']) - set(idcol) );
    numeric_cols = list( set(covariates._get_numeric_data().columns) - set(ordinal_cols) - set(irrelevant_cols)  );
    cat_cols = list( set(all_cols) - set(numeric_cols) - set(ordinal_cols) - set(irrelevant_cols) );
    if idcol in numeric_cols:
        numeric_cols.remove(idcol);
    elif idcol in cat_cols:
        cat_cols.remove(idcol);                                             
    
    # Drop columns that are not to be used at all
    if len(irrelevant_cols) > 0:
        covariates = covariates.drop(irrelevant_cols, axis=1);    
    
    print 'Applying labelizer and binarizer for categorical columns...';
    # Create a label encoder for all categorical covariates for later use.     
    if len(ordinal_cols) > 0:
        lblenc_dict = labelizer(covariates, ordinal_cols); 
        # Apply labelizer to the categorical columns in the dataset
        covariates = covariates[ordinal_cols].apply(lambda x: lblenc_dict[x.name].transform(x)).join(covariates[covariates.columns.difference(ordinal_cols)]);    
    # Apply binarizer to dataset for a one-hot encoding of dataset
    if len(cat_cols) > 0:
        covariates = binarizer(covariates, cat_cols);    
    
    # Create train, test and valid datasets for label and binary formats
    X_train = covariates.loc[covariates['rowtype'] == "TRAIN"].drop([idcol, "rowtype"], axis=1);
    X_valid = covariates.loc[covariates['rowtype'] == "VALID"].drop([idcol, "rowtype"], axis=1);
    X_test = covariates.loc[covariates['rowtype'] == "TEST"].drop([idcol, "rowtype"], axis=1);    
   
    # Post transformation dataset summaries.
    print 'Collecting post-transformation variable stats...' ;
    descstats(X_train.join(ycat)).to_csv("../output/traindata_stats.csv");
    descstats(X_train.join(ycat), ycat.name ).to_csv("../output/traindata_grouped_stats.csv"); 
    
    # Plot detailed histograms of variables    
    print 'Creating Histograms for numeric variables...';
    for col in numeric_cols:
        if not(col == ycol):
            plotdetailedhistogram(X_train.join(ycat), col, ycat.name );  
    
    print 'Completed data processing.';
    
    