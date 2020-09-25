########################################################################################################################
import matplotlib
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Manager
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score

from Classification_scripts.utils import split_group_cv, make_sure_folder_exists
from Classification_scripts import config
matplotlib.use("PDF")
num_cores = multiprocessing.cpu_count()
'''
This file to be used only in case of Classification
'''
########################################################################################################################

def classifiers():
    # List of classifiers to be used:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from xgboost import XGBClassifier

    classifiers = [(LogisticRegression(penalty=config.LRPenalty), 'LogisticRegression'),
                   (RandomForestClassifier(n_estimators=config.RFnEstim,
                                           n_jobs=config.nJobs,
                                           random_state=config.randstate), "RandomForest"),
                   (GradientBoostingClassifier(n_estimators=config.GBnEstim,
                                               random_state=config.randstate), "GradientBoost"),
                   (ExtraTreesClassifier(n_estimators=config.ETnEstim,
                                         random_state=config.randstate), "ExtraTrees"),
                   (DecisionTreeClassifier(random_state=config.randstate), "DecisionTrees"),
                   (BaggingClassifier(n_estimators=config.BCnEstim,
                                      n_jobs=config.nJobs,
                                      random_state=config.randstate), "Bagging"),
                   (AdaBoostClassifier(n_estimators=config.ABnEstim,
                                       random_state=config.randstate), "AdaBoost"),
                   (XGBClassifier(n_estimators=config.XGBnEstim,
                                  n_jobs=config.nJobs,
                                  randomstate=config.randstate), "XGBoost")
                   ]
    return classifiers

def cross_validate_and_plot(clf, X, y, column_names, name, splits, df):
    num_folds = splits

    # cv = StratifiedKFold(n_splits = num_folds)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    N, P = X.shape

    # Aggregate the importances over folds here:
    importances_random = np.zeros(P)
    importances_drop = np.zeros(P)

    # Loop over crossvalidation folds:

    scores = []  # Collect accuracies here

    tnList = []
    fpList = []
    fnList = []
    tpList = []
    precisionList = []
    #    recallList = []
    f1List = []
    mccList = []

    i = 1
    count = 0
    for train, test in split_group_cv(df, splits):
        count += 1

        #       print("Fitting model on fold %d/%d..." % (i, num_folds))

        X_train = X[train, :]
        y_train = y[train]
        X_test = X[test, :]
        y_test = y[test]

        # Train the model
        clf.fit(X_train, y_train)

        # Predict for validation data:
        probas_ = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # calculate confusion matrix, precision, f1 and Matthews Correlation Coefficient

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = precision_score(y_test, y_pred)
        #        recall = recall_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        tnList.append(tn / (tn + fp))
        tpList.append(tp / (fn + tp))
        fpList.append(fp / (tn + fp))
        fnList.append(fn / (fn + tp))

        precisionList.append(precision)
        #        recallList.append(recall)
        f1List.append(f1)
        mccList.append(mcc)
        #        print(classification_report(y_test, y_pred))

        # Finally: measure feature importance for each column at a time

        baseline = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        scores.append(baseline)

        ####### Importance calculated random shuffling column's elements
        for col in range(P):
            #            print("Assessing feature %d/%d..." % (col+1, P), end = "\r")

            # Store column for restoring it later
            save = X_test[:, col].copy()

            # Randomly shuffle all values in this column
            X_test[:, col] = np.random.permutation(X_test[:, col])

            # Compute AUC score after distortion
            m = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

            # Restore the original data
            X_test[:, col] = save

            # Importance is incremented by the drop in accuracy:
            importances_random[col] += (baseline - m)

        # save feature importance values in .csv
        idx = np.argsort(importances_random)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_random = pandas.DataFrame({'Variables': sorted_column_names, 'Importance': importances_random[idx]})
        target_file = "Importances_random_folds/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_random.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

        ####### Importance calculated dropping columns

        # create list shared by processes
        manager = Manager()
        n = manager.list()

        #        n=[]

        # define drop-column function
        def drop_column(col):
            # Drop a column at the time and fit the model
            X_drop = np.delete(X, np.s_[col], axis=1)
            X_train_drop = X_drop[train, :]
            X_test_drop = X_drop[test, :]
            clf.fit(X_train_drop, y_train)

            # Compute AUC score after distortion
            m = roc_auc_score(y_test, clf.predict_proba(X_test_drop)[:, 1])

            #            # Restore the original data
            #            X_test[:, col] = save

            n.append(m)

        # run drop-column function in parallel
        Parallel(n_jobs=num_cores, verbose=10)(delayed(drop_column)(col) for col in range(P))

        # Importance is incremented by the drop in accuracy:
        for col in range(P):
            importances_drop[col] += (baseline - n[col])

        # save feature importance values in .csv
        idx = np.argsort(importances_drop)
        sorted_column_names = list(np.array(column_names)[idx])

        importance_drop = pandas.DataFrame({'Variables': sorted_column_names, 'Importance': importances_drop[idx]})
        target_file = "Importance_drop_folds/%s-%s" % (name, count)
        make_sure_folder_exists(target_file)
        importance_drop.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

        i += 1
        print("\n")

    # Average the metrics over folds

    print("confusion matrix " + str(name))
    tnList = 100 * np.array(tnList)
    tpList = 100 * np.array(tpList)
    fnList = 100 * np.array(fnList)
    fpList = 100 * np.array(fpList)
    precisionList = 100 * np.array(precisionList)
    f1List = 100 * np.array(f1List)
    mccList = 100 * np.array(mccList)

    # show metrics

    print("TN: %.02f %% ± %.02f %% - FN: %.02f %% ± %.02f %%" % (np.mean(tnList),
                                                                 np.std(tnList),
                                                                 np.mean(fnList),
                                                                 np.std(fnList)))
    print("FP: %.02f %% ± %.02f %% - TP: %.02f %% ± %.02f %%" % (np.mean(fpList),
                                                                 np.std(fpList),
                                                                 np.mean(tpList),
                                                                 np.std(tpList)))
    #    print("Precision: %.02f %% ± %.02f %%  Recall: %.02f %% ± %.02f %%" % (np.mean(precisionList),
    #                                                       np.std(precisionList),
    #                                                       np.mean(recallList),
    #                                                       np.std(recallList)))
    print(
        "Precision: %.02f %% ± %.02f %% - F1: %.02f %% ± %.02f %% - MCC: %.02f %% ± %.02f %%" % (np.mean(precisionList),
                                                                                                 np.std(precisionList),
                                                                                                 np.mean(f1List),
                                                                                                 np.std(f1List),
                                                                                                 np.mean(mccList),
                                                                                                 np.std(mccList)))
    # save metrics as .csv
    metrics_tosave = pandas.DataFrame(
        {'Confusion_Matrix_TNR_mean': np.mean(tnList), 'Confusion_Matrix_TNR_std': np.std(tnList),
         'Confusion_Matrix_FNR_mean': np.mean(fnList), 'Confusion_Matrix_FNR_std': np.std(fnList),
         'Confusion_Matrix_FPR_mean': np.mean(fpList), 'Confusion_Matrix_FPR_std': np.std(fpList),
         'Confusion_Matrix_TPR_mean': np.mean(tpList), 'Confusion_Matrix_TPR_std': np.std(tpList),
         'Confusion_Matrix_Precision_mean': np.mean(precisionList),
         'Confusion_Matrix_Precision_std': np.std(precisionList),
         'Confusion_Matrix_F1_mean': np.mean(f1List), 'Confusion_Matrix_F1_std': np.std(f1List),
         'Confusion_Matrix_MCC_mean': np.mean(mccList), 'Confusion_Matrix_MCC_std': np.std(mccList)}, index=[0])

    target_file = "Metrics/%s" % name
    make_sure_folder_exists(target_file)
    metrics_tosave.to_csv(target_file + ".csv", columns=['Confusion_Matrix_TNR_mean', 'Confusion_Matrix_TNR_std',
                                                         'Confusion_Matrix_FNR_mean', 'Confusion_Matrix_FNR_std',
                                                         'Confusion_Matrix_FPR_mean', 'Confusion_Matrix_FPR_std',
                                                         'Confusion_Matrix_TPR_mean', 'Confusion_Matrix_TPR_std',
                                                         'Confusion_Matrix_Precision_mean',
                                                         'Confusion_Matrix_Precision_std',
                                                         'Confusion_Matrix_F1_mean', 'Confusion_Matrix_F1_std',
                                                         'Confusion_Matrix_MCC_mean', 'Confusion_Matrix_MCC_std'],
                          sep=';', index=False)

    # Average the TPR over folds

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot AUC curve

    plt.figure(1)
    plt.clf()

    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    target_file = "AUCs/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    # Plot importances:
    plt.figure(2)
    plt.clf()

    # Divide importances by num folds to get the average

    importances_average_random = importances_random / num_folds

    idx = np.argsort(importances_average_random)
    sorted_column_names = list(np.array(column_names)[idx])

    importance_average_random = pandas.DataFrame(
        {'Variables': sorted_column_names, 'Importance': importances_average_random[idx]})
    target_file = "Importances_random/values/%s" % name
    make_sure_folder_exists(target_file)
    importance_average_random.to_csv(target_file + ".csv", columns=['Variables', 'Importance'], sep=';', index=False)

    fontsize = 2 if P > 100 else 8

    plt.barh(np.arange(P), 100 * importances_average_random[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score [%])")
    plt.title("Feature importances (baseline AUC = %.2f %%)" % \
              (100 * np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_random/plots/%s" % name
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

    plt.barh(np.arange(P), 100 * importances_average_drop[idx], align='center')
    plt.yticks(np.arange(P), sorted_column_names, fontsize=fontsize)
    plt.xlabel("Feature importance (drop in score [%])")
    plt.title("Feature importances - Drop-column (baseline AUC = %.2f %%)" % \
              (100 * np.mean(scores)))

    plt.ylabel("<-- Less important     More important -->")

    target_file = "Importances_drop/plots/%s" % name
    make_sure_folder_exists(target_file)
    plt.savefig(target_file + ".pdf", bbox_inches="tight")

    return mean_fpr, mean_tpr
