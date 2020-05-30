# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:52:21 2019

@author: RV
"""

#%%

#%%
def confusionMatrixInfo(p,a, labels = None):
    #
    from sklearn.metrics import confusion_matrix as skm_conf_mat
#    p = pd.Series([1,1,1,0,0,0,0,0,0,0])
#    a = pd.Series([1,0,0,1,1,1,0,0,0,0])
#    labels = [1,0]
#
#    x = skm.confusion_matrix(a,p,labels=labels)
    if 'sklearn' not in sys.modules:
        import sklearn
    x = skm_conf_mat(a,p, labels = labels)
    tp = x[0,0]
    tn = x[1,1]
    fp = x[1,0]
    fn = x[0,1]
    # tp, fp, fn, tn # test order
    
    tsensitivity = tp/(tp+fn)
    tspecificity = tn/(tn + fp)
    # no information rate?
    tnir = (tp + fn)/x.sum()
    tnir = max(tnir, 1-tnir)
    # accuracy
    taccuracy = (tp + tn)/x.sum()
    
    res = {'confusionMatrix':x,
           'accuracy': taccuracy,
           'no information rate': tnir,
           'sensitivity': tsensitivity,
           'specificity': tspecificity
           }
    return(res)
    
#%% Test
"""
import pandas as pd
import numpy as np
import sklearn.metrics as skm

# Define two easy to understand series; a = actual (real), p = prediction/predicted
p = pd.Series([1,1,1,0,0,0,0,0,0,0])
a = pd.Series([1,0,0,1,1,1,0,0,0,0])
# assume 1 = True, 0 = False
# so we expect TP = 1, FP = 2, FN = 3, TN = 4
y = confusionMatrixInfo(p,a, labels=[1,0])
y
y['accuracy']
y['no information rate']
confusionMatrixInfo([1,1,1],[1,1,1]) # we get an error - why?
skm.confusion_matrix([1,1,1],[1,1,1])
skm.confusion_matrix([1,1,1],[1,1,1], labels=[1,0])
confusionMatrixInfo([1,1,1],[1,1,1], labels=[1,0]) # we get an error - why?
"""