# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:50:00 2019

@author: RV
"""
import numpy as np

XS = pd.read_pickle(os.path.join(savedDataFld, 'trainPredictorsDummies.pkl'))
y = pd.read_pickle(os.path.join(savedDataFld, 'trainOutcome.pkl'))

# KNN
# annoyingly (but justifiably) it requires testing dataset to make predictions
import sklearn as skl
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn

#%% build multiple models, compare accuracy
def knnOptimization(cmMetric = 'accuracy'):
    kvals = [i for i in range(1,15)]
    kmodels = {}
    kpreds = {}
    kpreds_prob = {}
    cutoffgrid = np.linspace(0,1,100)
    numk = []
    for k in kvals:
        tknn = knn(n_neighbors= k).fit(XS,y)
        kmodels[k] = tknn
        
        tknn_preds = tknn.predict(XS)
        kpreds[k] = tknn_preds
        
        tknn_preds_prob = tknn.predict_proba(XS)
        kpreds_prob[k] = [x[0] for x in tknn_preds_prob]
    for k in kvals:
        tcm = [confusionMatrixInfo(kpreds_prob[k] < i, y, labels=[1,0])[cmMetric] for i in cutoffgrid]
        numk.append(max(tcm))
        count = 0
        for x in numk:
            count +=1 
            if x == max(numk):
                return count, numk
print(knnOptimization('accuracy'))
print(knnOptimization('sensitivity'))      
print(knnOptimization('specificity'))  
        
KNN = knn(n_neighbors = 1).fit(XS, y)
    


