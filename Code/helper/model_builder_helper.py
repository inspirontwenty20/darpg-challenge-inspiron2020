#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split,cross_val_score # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Adding a step to convert the string lables to encoded values in dataframe.
def get_encoded_values(x):
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(x)
    return label_encoder.transform(x),label_encoder


def get_decision_tree_model(X_train,y_train):
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    return clf

def get_randomforest_model(X_train,y_train):
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    #Create a Gaussian Classifier
    #reducing the estimators to 10 from 1000 as our data size is too small. (As of now)
    clf=RandomForestClassifier(n_estimators=1000)
    clf = clf.fit(X_train,y_train)
    return clf

def get_prediction(clf,X):
     #Predict the response for test dataset
    return clf.predict(X) #getting error here. please check.

def get_metrics(y_pred,y_actual,message=""):
    #Metrics 
    #print("Cross validation",cross_val_score(clf, X, y, cv=10))
    print(message + " Accuracy:",metrics.accuracy_score(y_actual, y_pred))
    print(message + " F1 score:",metrics.f1_score(y_actual, y_pred, average='weighted'))

#tried implementing AUC_ROC, TPR and FPR and AUC Score, but 
#they are a bit trcky to implement for multiclass problem. 
# whie studying, i found about PyCM. Implmented that below
def get_auc_roc(model, X_test, y_test):
    from sklearn.metrics import roc_auc_score
    #roc_value = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    predict = model.predict(X_test)
    print(predict)
    #roc_value = roc_auc_score(y_test, model.predict_proba(X_test))
    roc_value = roc_auc_score(y_test, predict_probs)
    return roc_value

def get_false_true_positive_rates(model, X_test, y_test):
    from sklearn.metrics import roc_curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict(X_test))
    return false_positive_rate, true_positive_rate, thresholds

def auc_score(false_positive_rate, true_positive_rate):
    return auc(false_positive_rate, true_positive_rate)

def get_confusion_metric(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_test, y_pred)

###################################################################################
# Implementing PyCM here.                                                        ##
# PyCM is a multi-class confusion matrix library written in Python that support  ##
# both input data vectors and direct matrix, and a proper tool for               ##
# post-classification model evaluation that supports most classes and overall    ## 
# statistics parameters. PyCM is the swiss-army knife of confusion matrices,     ##
# targeted mainly at data scientists that need a broad array of metrics for      ##
# predictive models and an accurate evaluation of large variety of classifiers.  ##
# Read the documentation here: https://www.pycm.ir/doc/                          ## 
###################################################################################
from pycm import *
def get_PyCM(y_test, pred_test):
    
    import numpy as np
    cm = ConfusionMatrix(actual_vector=y_test.to_numpy(), predict_vector=pred_test) # Create CM From Data
    
    #Printing the classes in the problem
    print("Classes in y_test:", cm.classes)
    
    #class wise statistics. This is big print so commenting
    # Also most of the valus is givem in the below print. 
    #print(cm.class_stat)
    
    #printing the entire confuion matrix oject
    # this has a lot of information, but can be limited. 
    # see the documentation at: https://www.pycm.ir/doc/  
    print("Prnting the Confusion MAtrix and other stats: ")
    print(cm)
    return cm

