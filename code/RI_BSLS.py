#  -*- coding: utf-8 -*-
# Date: 2020
# Author: Yanzhe Kang <kyz1994@tju.edu.cn>
# Licence: GPL
# Desc: Reject inference using borderline smote and label spreading algorithm

import csv
import warnings
import time
import itertools
import pandas as pd
import numpy as np
from scipy import interp
from math import isnan
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, cohen_kappa_score, \
                            log_loss, brier_score_loss, hinge_loss, classification_report
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def baseline_resampling_labelspreading(data_path, bad_sample_num, good_sample_num, reject_sample_num, random_state_for_each_epoch, classifier, resampling_model):
    
    warnings.filterwarnings("ignore")
    raw_data_train = pd.read_csv(data_path, index_col='ID')

    data_bad = raw_data_train[raw_data_train['label'] == 1]
    # print data_bad.shape
    data_good = raw_data_train[(raw_data_train['label'] == 0)]
    data_reject = raw_data_train[raw_data_train['label'] == -1]

    """good 和 bad 采样 """
    data_bad_sampling = data_bad.sample(n=bad_sample_num, random_state=random_state_for_each_epoch)
    data_good_sampling= data_good.sample(n=good_sample_num, random_state=random_state_for_each_epoch)
    data_train = pd.concat([data_bad_sampling, data_good_sampling], axis=0)
    # print("All Data Size:" + str(data_train.shape))
    feature_name = list(data_train.columns.values)
    # print(feature_name)

    s = 0  
    np.random.seed(s)
    sampler = np.random.permutation(len(data_train.values))
    data_train_randomized = data_train.take(sampler)

    '''获得训练集的输入X和y'''
    y = data_train_randomized['label'].as_matrix()  
    X = data_train_randomized.drop(['label'], axis=1).as_matrix()  

    '''Split train/test data sets'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

    X_train_resampled, y_train_resampled = resampling_model.fit_sample(X_train, y_train)

    data_reject_sampling = data_reject.sample(n=reject_sample_num, random_state=random_state_for_each_epoch)
    y_reject = data_reject_sampling['label'].as_matrix()
    X_reject= data_reject_sampling.drop(['label'], axis=1)  

    X_train_resampled_and_reject = np.r_[X_train_resampled, X_reject]
    y_train_resampled_and_reject = np.r_[y_train_resampled, y_reject]

    ls_semi = LabelSpreading(kernel='rbf', gamma=5, alpha=0.5, max_iter=100, tol=0.001, n_jobs=-1)
    # ls_semi = LabelSpreading(kernel='rbf', gamma=2, alpha=0.5, max_iter=100, tol=0.001, n_jobs=-1)
    # ls_semi = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.5, max_iter=50, tol=0.001, n_jobs=-1)  # 
    y_reject_predict = ls_semi.fit(X_train_resampled_and_reject, y_train_resampled_and_reject).predict(X_reject)

    y_train_resampled_and_reject_1 = np.r_[y_train_resampled, y_reject_predict]

    # X_train_resampled_and_reject = np.r_[X_train, X_reject]  # 在good和bad构成的train set中加入reject的数据
    # y_train_resampled_and_reject = np.r_[y_train, y_reject_predict]
    # print(X_train_and_reject.shape)
    # print(y_train_and_reject.sum())

    '''Supervised Learning'''
    y_proba = classifier.fit(X_train_resampled_and_reject, y_train_resampled_and_reject_1).predict_proba(X_test)
    y_predict = classifier.fit(X_train_resampled_and_reject, y_train_resampled_and_reject_1).predict(X_test)

    # y_predict = y_proba[:, 1].copy()
    # y_predict[y_predict >= 0.9] = 1
    # y_predict[y_predict < 0.9] = 0

    '''Accuracy'''
    accuracy_result = accuracy_score(y_test, y_predict)
    # print("Accuracy Score:" + str(accuracy_result))

    '''Precision'''
    precision_result = precision_score(y_test, y_predict)
    # print("Precision Score:" + str(precision_result))

    '''Recall'''
    recall_result = recall_score(y_test, y_predict)
    # print("Recall Score:" + str(recall_result))

    '''F1'''
    f1_result = f1_score(y_test, y_predict)
    # print("F1 Score:" + str(f1_result))

    '''Log loss'''
    log_loss_result = log_loss(y_test, y_proba[:, 1])
    # print("logloss Score:" + str(log_loss_result))

    '''Cohen-Kappa'''
    cohen_kappa_result = cohen_kappa_score(y_test, y_predict)
    # print("Cohen-Kappa Score:" + str(cohen_kappa_result))

    '''brier score'''
    brier_result = brier_score_loss(y_test, y_predict)
    # print("brier Score:" + str(brier_result))

    '''AUC and ROC curve'''
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    auc_result = auc(fpr, tpr)
    # print("AUC Score:" + str(auc_result))

    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    '''K-S Value'''
    ks_result = max(tpr - fpr)

    '''Classification Report'''
    # target_names = ['class 0', 'class 1', 'class 2']
    # print(classification_report(y_test, y_predict, target_names=target_names))

    '''Confusion Matrix'''
    # # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_predict)
    # np.set_printoptions(precision=2)
    #
    # # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix, without normalization')
    #
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=[0, 1], normalize=True, title='Normalized confusion matrix')
    #
    # plt.show()

    # print("Accuracy Score:" + str(accuracy_result) + " Precision Score:" + str(precision_result) + " Recall Score:" + str(recall_result) +
    #       " F1 Score:" + str(f1_result) + " logloss Score:" + str(log_loss_result) + " Cohen-Kappa Score:" + str(cohen_kappa_result) +
    #       " brier Score:" + str(brier_result) + " AUC Score:" + str(auc_result))

    return accuracy_result, precision_result, recall_result, f1_result, log_loss_result, cohen_kappa_result, brier_result, ks_result, auc_result


