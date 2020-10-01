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

from sklearn.metrics import auc

import time
import pickle

from Classification_scripts.utils import create_groups_cv
from Classification_scripts.Classification_crossValidation import cross_validate_and_plot, classifiers
from Classification_scripts import config

matplotlib.use("PDF")
########################################################################################################################
start_time = time.time()

if __name__ == "__main__":
    # Read in data and create the variable df to manipulate it
    df = pandas.read_csv(config.CSVOrigin)
    df = create_groups_cv(df, config.nFolds)

    # create the two other columns - TO CANCEL OR MODIFY - IT IS MODEL SPECIFIC
    # df['M_functions_to_classes'] = df['M_functions'] / df['M_classes']
    # df['M_duplicatelines_to_ncloc'] = df['M_duplicated_lines'] / df['M_ncloc']

    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # variables assignement
    y = df[config.target]
    colX = [c for c in df.columns if "squid" in c]
    # colsM = [c for c in df.columns if "M_" in c]
    # colsSM = sum([cols, colsM], [])

    # variable generation for tests
    X = df[colX]
    # XM = df[colsM]
    # XSM = df[colsSM]
    X = np.array(X)
    # XM = np.array(XM)
    # XSM = np.array(XSM)
    y = np.array(y)

    if config.analysisType == 'Classification':
        from Classification_scripts.Classification_crossValidation import cross_validate_and_plot, classifiers
        model = classifiers()
    if config.analysisType == 'Regression':
        from Classification_scripts.Regression_crossValidation import cross_validate_and_plot, regressors
        model = regressors()

    # # Loop over each and cross-validate
    # clf_to_use = config.loops

    ################################################# SQUID Prediction ################################c#################
    if config.analysisType == 'Classification':
        for clf, name in model:
            # clf, name = model[clf_to_use]
            print("Evaluating %s classifier (squid)" % name)
            fpr, tpr = cross_validate_and_plot(clf, X, y, colX, name + "_squid", config.nFolds, df)
            squid_rocs = [name, fpr, tpr, auc(fpr, tpr)]

            with open('squid_rocs_%s.data' % name, 'wb') as fp:  # Pickling
                pickle.dump(squid_rocs, fp)

    if config.analysisType == 'Regression':

        for clf, name in model:
            print("Evaluating %s classifier (squid)" % name)
            mae, r2 = cross_validate_and_plot(clf, X, y, colX, name + "_squid", config.nFolds, df)
            squid_metrics = [name, mae, r2]

            with open('squid_changes_%s.data' % name, 'wb') as fp:  # Pickling
                pickle.dump(squid_metrics, fp)

    # ################################################### M Prediction ###################################################
    #
    # # for clf, name in classifiers:
    # clf, name = model[clf_to_use]
    # print("Evaluating %s classifier (M)" % name)
    # fpr, tpr = cross_validate_and_plot(clf, XM, y, colsM, name + "_M", config.nFolds, df)
    # M_rocs = [name, fpr, tpr, auc(fpr, tpr)]
    # with open("M_rocs_%s.data" % name, "wb") as fp:  # Pickling
    #     pickle.dump(M_rocs, fp)
    #
    # ############################################### SQUID + M Prediction ###############################################
    #
    # # for clf, name in classifiers:
    # clf, name = model[clf_to_use]
    # print("Evaluating %s classifier (Squid + M)" % name)
    # fpr, tpr = cross_validate_and_plot(clf, XSM, y, colsSM, name + "_squid-M", config.nFolds, df)
    # squid_M_rocs = [name, fpr, tpr, auc(fpr, tpr)]
    # with open("squid_M_rocs_%s.data" % name, "wb") as fp:  # Pickling
    #     pickle.dump(squid_M_rocs, fp)

##################################################### End of Main ######################################################
# time info
end_time = time.time()
execution_time = end_time - start_time
print("Total execution time: %s seconds" % execution_time)
