
"""
version 2.0
@author: Sergio Moreschini, Francesco Lomio

This is a config file: it means all the parameter that can be changed are stored and configured in this .py file
"""
################################################## General parameters ##################################################
# Address of the file to import to read the data
CSVOrigin = "../Data/TOTAL_finalDelta.csv"

analysisType = 'Regression' #choose 'Regression' or 'Classification'

target = 'isInducing 0/1?' #name of target variable

# n-folds for cross-validation
nFolds = 10

# # number of loops for cross validation
# loops = 5
#################################################### Models dependant ####################################################
# General Settings
nJobs = -1 #-1 uses all cores available
randstate = 0 #define random state for models initialization

# 1) Logistic Regression
#    penalty for LR
LRPenalty = "l2" #no L1


# 2) Random Forest
#    number of estimators
RFnEstim = 100


# 3) Gradient Boosting
#    number of estimators
GBnEstim = 100


# 4) Extra Trees
#    number of estimators
ETnEstim = 100


# 5) Decision Tree


# 5) Bagging Classifier
#    number of estimators
BCnEstim = 100


# 6) AdaBoost
#    number of estimators
ABnEstim = 100


# 7) XGB classifier
#    number of estimators
XGBnEstim = 100

########################################################################################################################
