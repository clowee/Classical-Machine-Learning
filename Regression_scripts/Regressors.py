# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("PDF")

import os
import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_score
# from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Manager

from scipy import interp
# from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn.model_selection import StratifiedKFold

import time
import pickle
import json

from validation import split_group_cv, create_groups_cv

num_cores = multiprocessing.cpu_count()

def cross_validate_and_plot(clf, X, y, column_names, name, splits):
    num_folds = splits

    # cv = StratifiedKFold(n_splits = num_folds)
    #    mean_fpr = np.linspace(0, 1, 100)
    #    tprs = []
    #    aucs = []

    N, P = X.shape

    # Aggregate the importances over folds here:
    importances_random = np.zeros(P)
    importances_drop = np.zeros(P)

    # Loop over crossvalidation folds:

    scores = []  # Collect accuracies here

    R2 = []
    MAE = []

    i = 1
    count = 0
    for train, test in split_group_cv(df, splits):
        count += 1

        # print("Fitting model on fold %d/%d..." % (i, num_folds))

        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]

        # Train the model
        clf.fit(X_train, y_train)

        # Predict for validation data:
        #        probas_ = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        # Compute MAE and R2 score
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        R2.append(r2)
        MAE.append(mae)

        # Finally: measure feature importances for each column at a time

        baseline = mean_absolute_error(y_test, clf.predict(X_test))
        scores.append(baseline)

        ####### Importance calculated random shuffling column's elements
        for col in range(P):
            #            print("Assessing feature %d/%d..." % (col+1, P), end = "\r")

            # Store column for restoring it later
            save = X_test[:, col].copy()

            # Randomly shuffle all values in this column
            X_test[:, col] = np.random.permutation(X_test[:, col])

            # Compute AUC score after distortion
            m = mean_absolute_error(y_test, clf.predict(X_test))

            # Restore the original data
            X_test[:, col] = save

            # Importance is incremented by the drop in accuracy:
            importances_random[col] += -(baseline - m)

        # save feature importance values in .csv
        idx = np.argsort(importances_random)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_random = pandas.DataFrame({'Variables': sorted_column_names, 'Importance': importances_random[idx]})
        target_file = "Importances_random_folds_Change/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_random.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

        ####### Importance calculated dropping columns

        # create list shared by processes
        manager = Manager()
        n = manager.list()

        # define drop-column function
        def drop_column(col):
            # Drop a column at the time and fit the model
            X_drop = np.delete(X, np.s_[col], axis=1)
            X_train_drop = X_drop[train, :]
            X_test_drop = X_drop[test, :]
            clf.fit(X_train_drop, y_train)

            # Compute AUC score after distortion
            m = mean_absolute_error(y_test, clf.predict(X_test_drop))


            n.append(m)

        # run drop-column function in parallel
        Parallel(n_jobs=num_cores, verbose=10)(delayed(drop_column)(col) for col in range(P))

        # Importance is incremented by the drop in accuracy:
        for col in range(P):
            importances_drop[col] += -(baseline - n[col])

        # save feature importance values in .csv
        idx = np.argsort(importances_drop)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_drop = pandas.DataFrame({'Variables': sorted_column_names, 'Importance': importances_drop[idx]})
        target_file = "Importances_drop_folds_Change/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_drop.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

        i += 1
        print("\n")

    mean_mae = np.mean(MAE)
    mean_r2 = np.mean(R2)

    metrics_tosave = pandas.DataFrame(
        {'Regressor': name, 'MAE': np.mean(MAE), 'MAE_std': np.std(MAE),
         'R2': np.mean(R2), 'R2_std': np.std(R2)}, index=[0])

    target_file = "Metrics_Change/%s" % name
    make_sure_folder_exists(target_file)
    metrics_tosave.to_csv(target_file + ".csv", columns=['Regressor', 'MAE', 'MAE_std',
                                                         'R2', 'R2_std'], index=False)


    # Plot importances:
    plt.figure(2)
    plt.clf()

    # Divide importances by num folds to get the average

    importances_average_random = importances_random / num_folds

    idx = np.argsort(importances_average_random)
    sorted_column_names = list(np.array(column_names)[idx])

    importance_average_random = pandas.DataFrame(
        {'Variables': sorted_column_names, 'Importance': importances_average_random[idx]})
    target_file = "Importances_random_Change/values/%s" % name
    make_sure_folder_exists(target_file)
    importance_average_random.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

    fontsize = 2 if P > 100 else 8

    plt.barh(np.arange(P), importances_average_random[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score )")
    plt.title("Feature importances (baseline MAE = %.2f)" % \
              (np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_random_Change/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    ######## Plot importance with dropping:
    # Plot importances:
    plt.figure(2)
    plt.clf()

    # Divide importances by num folds to get the average

    importances_average_drop = importances_drop / num_folds

    idx = np.argsort(importances_average_drop)
    sorted_column_names = list(np.array(column_names)[idx])

    importance_average_drop = pandas.DataFrame(
        {'Variables': sorted_column_names, 'Importance': importances_average_drop[idx]})
    target_file = "Importances_drop/values/%s" % name
    make_sure_folder_exists(target_file)
    importance_average_drop.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

    fontsize = 2 if P > 100 else 8

    plt.barh(np.arange(P), importances_average_drop[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score)")
    plt.title("Feature importances - Drop-column (baseline MAE = %.2f)" % \
              (np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_drop_Change/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    return mean_mae, mean_r2


def make_sure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)



start_time = time.time()

if __name__ == "__main__":

    #define number of split for the cross validation
    splits = 10
    # Read in data

    # read data
    df = pandas.read_csv("JSS2018.csv", nrows=100)

    df = create_groups_cv(df, splits)

    # create the two other columns
    # df['M_functions_to_classes'] = df['M_functions']/df['M_classes']
    # df['M_duplicatelines_to_ncloc'] = df['M_duplicated_lines']/df['M_ncloc']
    # remove infinite values and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = df["Git_CodeChurns"] # select target variable
    cols = [c for c in df.columns if "ADDED_SQUID" in c] #select input variable
    # cols.append("Groups")
    # colsM = [c for c in df.columns if "M_" in c]
    # colsM.append("Groups")
    # colsSM = sum([cols, colsM], [])
    # colsSM.append("Groups")

    X = df[cols]

    X = np.array(X)

    y = np.array(y)

    # Define classifiers to try: (clf, name) pairs

    classifiers = [(LinearRegression(n_jobs=-1), 'LinearRegression'),
                   (RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0), "RandomForest"),
                   (GradientBoostingRegressor(n_estimators=100, random_state=0), "GradientBoost"),
                   (ExtraTreesRegressor(n_estimators=100, random_state=0), "ExtraTrees"),
                   (DecisionTreeRegressor(random_state=0), "DecisionTrees"),
                   (BaggingRegressor(n_estimators=100, n_jobs=-1, random_state=0), "Bagging"),
                   (AdaBoostRegressor(n_estimators=100, random_state=0), "AdaBoost")
        # ,
        #            (XGBRegressor(n_estimators=100, n_jobs=-1, randomstate=0), "XGBoost")
                   ]


    ######## SQUID Prediction

    # Store all ROC curves here:
    squid_rocs = []

    for clf, name in classifiers:
        print("Evaluating %s classifier (squid)" % name)
        mae, r2 = cross_validate_and_plot(clf, X, y, cols, name + "_squid", splits)
        squid_rocs = [name, mae, r2]

    with open('squid_changes_%s.data' % name, 'wb') as fp:  # Pickling
        pickle.dump(squid_rocs, fp)


end_time = time.time()
execution_time = end_time - start_time
print("Total execution time: %s seconds" % execution_time)
