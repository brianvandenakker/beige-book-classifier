# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 00:14:38 2019

@author: RV
Purpose: use Ridge, Lasso and Elastic Net models on Titanic data
"""
X = pd.read_pickle(os.path.join(savedDataFld, 'trainPredictorsDummies.pkl'))
y = pd.read_pickle(os.path.join(savedDataFld, 'trainOutcome.pkl'))

#%% Lasso
from sklearn.linear_model import Lasso
useAlpha = 0.00001

lasso = Lasso(alpha = useAlpha).fit(X, y)

[i for i in zip(X.columns, lasso.coef_)]

# predictions
lasso_is_pred = lasso.predict(X)

# attempt to identify a good cutoff
cutoffgrid = np.linspace(min(lasso_is_pred), max(lasso_is_pred), 100)

tcmLasso = [confusionMatrixInfo(lasso_is_pred < i, y, labels=[1,0])['accuracy'] for i in cutoffgrid]
cutoffLasso = max(tcmLasso)

#%%
# Ridge classifier
from sklearn.linear_model import RidgeClassifierCV as RCCV

RCCV = RCCV(alphas=[np.exp(i) for i in np.linspace(-10,0,50)]).fit(X,y)
RCCV.score(X,y) # not that great ?
RCCV_is_pred = RCCV.predict(X)
confusionMatrixInfo(RCCV_is_pred,y)

# attempt to identify a good cutoff
cutoffgrid = np.linspace(min(RCCV_is_pred), max(RCCV_is_pred), 100)

tcmRCCV = [confusionMatrixInfo(RCCV_is_pred < i, y, labels=[1,0])['accuracy'] for i in cutoffgrid]

#%%
from sklearn.linear_model import ElasticNet as ENet

a = 0.0001
b = 0.0001
alpha = a+b
l1_ratio = a/(a+b)

ENet = ENet(alpha = alpha, l1_ratio= l1_ratio).fit(X,y)

ENet_is_pred = ENet.predict(X)

cutoffgrid = np.linspace(min(ENet_is_pred), max(ENet_is_pred), 50)

tcmENet = [confusionMatrixInfo(ENet_is_pred < i, y, labels=[1,0])['accuracy'] for i in cutoffgrid]
cutoffENet = max(tcmENet)

