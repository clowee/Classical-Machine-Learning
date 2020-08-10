
"""
version 2.0
@author: Sergio Moreschini, Francesco Lomio

This is a config file: it means all the paramenter that can be changed are stored and configured in this .py file
"""
################################################## General parameters ##################################################
# Address of the file to import to read the data
CSVOrigin = "./Data/TOTAL_finalDelta.csv"

# n-folds for cross-validation
nFolds = 10

# number of loops for cross validation
loops = 5
#################################################### Case dependant ####################################################
# 1) Logistic Regression
#    penalty for LR
LRPenalty = "l1"


# 2) Random Forest
#    number of estimators
RFnEstim = 100
#    number of jobs
RFnJobs = -1
#    random state
RFrs = 0


# 3) Gradient Boosting
#    number of estimators
GBnEstim = 100
#    random state
GBrs = 0


# 4) Extra Trees
#    number of estimators
ETnEstim = 100
#    random state
ETrs = 0

# 5) Decision Tree
#    random state
DCrs = 0

# 5) Bagging Classifier
#    number of estimators
BCnEstim = 100
#    number of jobs
BCnJobs = -1
#    random state
BCrs = 0

# 6) AdaBoost
#    number of estimators
ABnEstim = 100
#    random state
ABRs = 0

# 7) XGB classifier
#    number of estimators
XGBnEstim = 100
#    number of jobs
XGBnJobs = -1
#    random state
XGBrs = 0

########################################################################################################################
