import os;
os.chdir("../Documents/code/ml_template/src");

from sklearn.cross_validation import StratifiedKFold;
from sklearn.cross_validation import KFold;
from pandas import read_csv;


def splitter(dataset, target, eval_size):    
    kf = StratifiedKFold(dataset[target], round(1./eval_size));
    train_indices, valid_indices = next(iter(kf));
    X_train, y_train = dataset.drop([target], axis=1)[train_indices], dataset[train_indices,target];
    X_valid, y_valid = dataset.drop([target], axis=1)[valid_indices], dataset[valid_indices,target];
    return X_train, y_train, X_valid, y_valid;
    
    
def __main__():
    dataset = read_csv("../data/train.csv");