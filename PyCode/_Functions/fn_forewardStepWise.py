# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:40:57 2019

@author: brian
Purpose: Use foreward stepwise selection and 10-fold cross validation to select 'best' predictors.
Inputs: Model of choice (Logit, MLR, LDA, QDA)
Outputs: Model with highest prediction accuracy.
"""
#%%
from sklearn.linear_model import LogisticRegression as LogReg
#Foreward Stepwise regression with 10-fold cross validation
def forewardStepWise(model = LogReg(random_state = 0, solver = 'lbfgs', max_iter = 1000)):
    pred = X.copy()
    usedVar = []
    nVarModel = {}
    optimalVar = []
    for step in range(1, len(X.columns) + 1):
        accuracySelector = []
        for i in pred.columns:
            usedVar.append(i)
            xModel = model.fit(X[usedVar],y)
            CVaccuracy = cross_val_score(xModel, X[usedVar], y, scoring = 'accuracy', cv=10).mean()
            accuracySelector.append(CVaccuracy)
            usedVar = usedVar[:-1]
        nStep = max(accuracySelector)
        optimalVar.append(nStep)
        for n in range(len(accuracySelector)):
            if accuracySelector[n-1] == nStep:
                locate = n-1
        usedVar.append(pred.columns[locate])
        pred = pred.drop(columns = {pred.columns[locate]})
        nVarModel[step] = usedVar  #I have NO idea why this dictionary is adding new strings to keys 
                                   #from (step-1) in addition to the correct string for the current step... .
    nVarModel = {key: value[:-1] for key, value in nVarModel.items()}
    nVarModel[step] = usedVar
    maxAccuracy = max(optimalVar)
    count = 0
    for i in optimalVar:
        count += 1
        if maxAccuracy == i:
            goodModel = nVarModel[count]
    return goodModel
#%%