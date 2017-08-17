# pyanalytics-reusablesnippets

 An attempt to create reusable snippets for a data independent, semi-automised pipeline for quick prototyping of machine learning models. Has a pipeline for both Python and R, and the intent is to create reusable components based on [this blog](http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/). 
 
 Current functionality includes the following:
 - automated creation of descriptive stats & histograms for both categorical and numeric variables in relation to the dependent variable, 
 - semi-automated feature selection using principal components analysis and ANOVA F value between label/feature
 - semiautomated boosted trees (xgboost) pipeline with cross-validation and grid search for hyperparameter tuning for both regression and classification tasks (only in Python)
 - a random forest one with the same functionality as above (still in progress).

## Requirements:
- Python 2.7
- scikit-learn 0.18

## Pending items:
- generalised linear model
- Neural network
- SVM (maybe..)
- An ensemble model with another layer of validation
