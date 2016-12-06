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
def plotdetailedhistogram(dataset, x, y, prefix=""):
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
    plt.savefig("../output/plots/" + prefix + "hist_" + x + ".png", dpi=500);        
    plt.close();

def categorical_barplots(dataset, x, y, prefix=""):
    gby_obj = dataset.groupby(y)[x].value_counts().sort_index().unstack();
    l = int(np.ceil(np.sqrt(len(gby_obj.columns))));
    gby_obj.plot(kind="bar", subplots=True, layout=(l,l));
    plt.savefig("../output/plots/" + prefix + "bar_" + x + ".png", dpi=500);        
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
    ordinal_cols=['cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat45', 'cat44', 'cat47', 'cat46', 'cat41', 'cat40', 'cat43', 'cat42',
 'cat49', 'cat48', 'cat56', 'cat57', 'cat54', 'cat55', 'cat52', 'cat53', 'cat50', 'cat51', 'cat58', 'cat59', 'cat114', 'cat115', 'cat116', 'cat110', 'cat111', 'cat112',
 'cat113', 'cat63', 'cat62', 'cat61', 'cat60', 'cat67', 'cat66', 'cat65', 'cat64', 'cat69', 'cat68', 'cat109', 'cat108', 'cat103', 'cat102', 'cat101', 'cat100', 
 'cat107', 'cat106', 'cat105', 'cat104', 'cat78', 'cat79', 'cat74', 'cat75', 'cat76', 'cat77', 'cat70', 'cat71', 'cat72', 'cat73', 'cat81', 'cat80', 'cat83', 'cat82',
 'cat85', 'cat84', 'cat87', 'cat86', 'cat89', 'cat88', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat92', 'cat93', 'cat90', 'cat91',
 'cat96', 'cat97', 'cat94', 'cat95', 'cat98', 'cat99', 'cat18', 'cat19', 'cat12', 'cat13', 'cat10', 'cat11', 'cat16', 'cat17', 'cat14', 'cat15', 'cat27', 'cat26',
 'cat25', 'cat24', 'cat23', 'cat22', 'cat21', 'cat20', 'cat29', 'cat28'];
    # 0 for binary classification, 1 for multinomial classification, 2 for regression
    prob_type = 2; 
    # variable for target column quantiles in case target variable is continuous(for grouped desriptive statistics creation)
    quantiles=4; 
    print 'Read completed.';
                        
    #########################################################################################    
    print 'Collecting metadata about dataset...';
    
    # Classify attributes in train data into numeric, categorical and ordinal
    all_cols = list( set(full_dataset.columns) - set(['rowtype']) - set(idcol) );
    numeric_cols = list( set(full_dataset._get_numeric_data().columns) - set(ordinal_cols) - set(irrelevant_cols)  );
    cat_cols = list( set(all_cols) - set(numeric_cols) - set(ordinal_cols) - set(irrelevant_cols) );
    if idcol in numeric_cols:
        numeric_cols.remove(idcol);
    elif idcol in cat_cols:
        cat_cols.remove(idcol);  
    
    # Split train dataset labels and covariates into different dataframes.
    print 'Splitting covariates and target...';
    y = full_dataset.loc[:, ycol];    
    X = full_dataset.drop([ycol], axis=1);

 
    #########################################################################################  
    # DATA CLEANING AND TRANSFORMATION, DESCRIPTIVE STATS
    print 'Applying cleaning rules and data transformations...';
    
    # Treat missing values  
    X = X.dropna(); 
    
    # Data Cleaning and Transformations
    # Add new variables
    # <PLACEHOLDER FOR NON-GENERIC CODE: INSERT CODE HERE>
    y = np.log(1+y);   
    # <PLACEHOLDER FOR NON-GENERIC CODE: INSERT CODE HERE>
    
    ##########################################################################################        
    # Set outcome variable to categorical if problem is classification, else to float64
    if prob_type == 0 or prob_type == 1:
        y = y.astype("category");
        ycat = y;        
    elif prob_type==2:
        y = y.astype("float64");  
        ycat = pd.qcut(y, quantiles); # Quantile based cuts for target column
        ycat.name = ycat.name + '_cat';
        
    # Get first iteration of the k-fold indices, use it for the train-validation split
    # Other iterations may be used later  
    #print 'Splitting training data into training and validation sets...';
    skf = StratifiedKFold(n_splits=int(1./validation_size), shuffle=True);
    skf.get_n_splits(X, y);
    train_indices, valid_indices = next(iter(skf.split(X, ycat)));
    # Scale the numeric columns if required.
    X = X.join(pd.Series('TRAIN', index=train_indices, name = 'rowtype').append(pd.Series('VALID', index=valid_indices, name = 'rowtype')));
    X_test=test_dataset.join(pd.Series('TEST', index=test_dataset.index, name = 'rowtype'));   
    
    
    # Combine train, valid and test covariates to create a consolidated covariate set
    covariates = pd.concat([X, X_test], axis=0, ignore_index=True);     
    # If id column does not exist, create one.
    if (idcol is None) or ( len(idcol) == 0 ):
        idcol = 'id';
        covariates=covariates.join(pd.Series( range(1, len(covariates) + 1,1), index=covariates.index, name = idcol ));
    
    # Find and add columns with zero std deviation to irrelevant columns- These add no information.                    
    irrelevant_cols = irrelevant_cols + (covariates.std(axis=0, numeric_only=True) < 0.5)[(covariates.std(axis=0) == 0.0)].index.tolist();                                                     
    
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
    y_train = y.iloc[train_indices];
    y_valid = y.iloc[valid_indices];
   
    # Post transformation dataset summaries.
    print 'Collecting post-transformation variable stats...' ;
    descstats(X_train.join(y).join(ycat)).to_csv("../output/traindata_stats.csv");
    descstats(X_train.join(ycat), ycat.name ).to_csv("../output/traindata_grouped_stats.csv"); 
    
    # Plot detailed histograms of variables    
    print 'Creating Histograms and bar plots for variables...';   
    for col in numeric_cols:
        plotdetailedhistogram(X_train.join(y).join(ycat), col, ycat.name );
    for col in ordinal_cols:
        categorical_barplots(X_train.join(ycat), col, ycat.name);
    for col in cat_cols:
        categorical_barplots(X_train.join(ycat), col, ycat.name);
    
    print 'Completed data processing.';
    

    
    