#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version 2.0
@author: Sergio Moreschini, Francesco Lomio

"""
########################################################################################################################
import matplotlib
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import auc

import time
import pickle

from Classification_scripts.utils import create_groups_cv
from Classification_scripts.crossValidation import cross_validate_and_plot
from Classification_scripts import config

matplotlib.use("PDF")
########################################################################################################################
start_time = time.time()

if __name__ == "__main__":
    # Read in data and create the variable df to manipulate it
    df = pandas.read_csv(config.CSVOrigin)
    df = create_groups_cv(df, config.nFolds)

    # create the two other columns
    df['M_functions_to_classes'] = df['M_functions'] / df['M_classes']
    df['M_duplicatelines_to_ncloc'] = df['M_duplicated_lines'] / df['M_ncloc']

    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # variables assignement
    y = df["isInducing 0/1?"]
    cols = [c for c in df.columns if "squid" in c]
    colsM = [c for c in df.columns if "M_" in c]
    colsSM = sum([cols, colsM], [])

    # variable generation for tests
    X = df[cols]
    XM = df[colsM]
    XSM = df[colsSM]
    X = np.array(X)
    XM = np.array(XM)
    XSM = np.array(XSM)
    y = np.array(y)

    # List of classifiers to be used:
    classifiers = [(LogisticRegression(penalty=config.LRPenalty), 'LogisticRegression'),
                   (RandomForestClassifier(n_estimators=config.RFnEstim, n_jobs=config.RFnJobs,
                                           random_state=config.RFrs), "RandomForest"),
                   (GradientBoostingClassifier(n_estimators=config.GBnEstim,
                                               random_state=config.GBrs), "GradientBoost"),
                   (ExtraTreesClassifier(n_estimators=config.ETnEstim, random_state=config.ETrs), "ExtraTrees"),
                   (DecisionTreeClassifier(random_state=config.DCrs), "DecisionTrees"),
                   (BaggingClassifier(n_estimators=config.BCnEstim, n_jobs=config.BCnJobs,
                                      random_state=config.BCrs), "Bagging"),
                   (AdaBoostClassifier(n_estimators=config.ABnEstim, random_state=config.ABRs), "AdaBoost"),
                   (XGBClassifier(n_estimators=config.XGBnEstim, n_jobs=config.XGBnJobs,
                                  randomstate=config.XGBrs), "XGBoost")
                   ]

    # Loop over each and cross-validate
    clf_to_use = config.loops

    ################################################# SQUID Prediction #################################################

    # for clf, name in classifiers:
    clf, name = classifiers[clf_to_use]
    print("Evaluating %s classifier (squid)" % name)
    fpr, tpr = cross_validate_and_plot(clf, X, y, cols, name + "_squid", config.nFolds, df)
    squid_rocs = [name, fpr, tpr, auc(fpr, tpr)]

    with open('squid_rocs_%s.data' % name, 'wb') as fp:  # Pickling
        pickle.dump(squid_rocs, fp)

    ################################################### M Prediction ###################################################

    # for clf, name in classifiers:
    clf, name = classifiers[clf_to_use]
    print("Evaluating %s classifier (M)" % name)
    fpr, tpr = cross_validate_and_plot(clf, XM, y, colsM, name + "_M", config.nFolds, df)
    M_rocs = [name, fpr, tpr, auc(fpr, tpr)]
    with open("M_rocs_%s.data" % name, "wb") as fp:  # Pickling
        pickle.dump(M_rocs, fp)

    ############################################### SQUID + M Prediction ###############################################

    # for clf, name in classifiers:
    clf, name = classifiers[clf_to_use]
    print("Evaluating %s classifier (Squid + M)" % name)
    fpr, tpr = cross_validate_and_plot(clf, XSM, y, colsSM, name + "_squid-M", config.nFolds, df)
    squid_M_rocs = [name, fpr, tpr, auc(fpr, tpr)]
    with open("squid_M_rocs_%s.data" % name, "wb") as fp:  # Pickling
        pickle.dump(squid_M_rocs, fp)

##################################################### End of Main ######################################################
# time info
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time: %s seconds" % execution_time)
