#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:48:02 2019

@author: apple
"""

#%% split up the files 
from sklearn.model_selection import train_test_split as skl_traintest_split
from sklearn.ensemble import GradientBoostingClassifier as GBClass

#%% Gradient boosting
params = {'n_estimators':100, 'subsample':1.0, 'learning_rate':0.1}
params = dict(params)
model_gbc = GBClass(**params)

#%%

bifurcateDigits(train, 0, train = True)   # when the digit is 0 
bifurcateDigits(test, 0, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]
​
confusionMatrixInfo(pred_Y, y_test, labels = None)

# RUC and AUC rate
fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
# plt.figure(1)
# plt.plot(fpr_gbc, tpr_gbc, 'r-')

# AUC 
skm.auc(fpr_gbc,tpr_gbc)       
#%% 

bifurcateDigits(train, value, train = True)   # when the digit is 1 
bifurcateDigits(test, value, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]
    
confusionMatrixInfo(pred_Y, y_test, labels = None)
    
fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%

bifurcateDigits(train, 2, train = True)   # when the digit is 2 
bifurcateDigits(test, 2, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 3, train = True)   # when the digit is 3 
bifurcateDigits(test, 3, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 4, train = True)   # when the digit is 4 
bifurcateDigits(test, 4, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 5, train = True)   # when the digit is 5 
bifurcateDigits(test, 5, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 6, train = True)   # when the digit is 6 
bifurcateDigits(test, 6, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 7, train = True)   # when the digit is 7 
bifurcateDigits(test, 7, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 8, train = True)   # when the digit is 8 
bifurcateDigits(test, 8, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
bifurcateDigits(train, 9, train = True)   # when the digit is 9 
bifurcateDigits(test, 9, train = False)
model_gbc.fit(X_train, y_train)
pred_Y = model_gbc.predict(X_test)
z_gbc = model_gbc.predict_proba(X_test)[:,1]

confusionMatrixInfo(pred_Y, y_test, labels = None)

fpr_gbc, tpr_gbc, thresh_gbc = skm.roc_curve(y_test, z_gbc)
skm.auc(fpr_gbc,tpr_gbc)

#%%
#Results Per Model

# AUC(digits 0-9) 
# 0:  0.9984546050087215,
# 1:  0.9990136690458005
# 2:  0.9930195552950043
# 3:  0.9921168806626702
# 4:  0.9935744481811293
# 5:  0.9888587281263337
# 6:  0.998162574290247
# 7:  0.9974924286627211
# 8:  0.9937558339264936
# 9:  0.9902653826553945

#Accuracy(digits 0-9)
# 0:  0.9830402010050251
# 1:  0.9868043602983362
# 2:  0.9616066154754873
# 3:  0.9610538373424972
# 4:  0.9628953771289538
# 5:  0.9477465708687133
# 6:  0.9773869346733668
# 7:  0.9781968179139658
# 8:  0.9630512514898689
# 9:  0.9493975903614458

#Average Accuracy: 0.9671
