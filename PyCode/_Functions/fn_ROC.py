# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:48:57 2019

@author: RV
Purpose: create ROC plotting function
Inputs: probabilities given by a model, one per records
        actual outcomes (Y/N) from the test data
Outputs: plot of ROC curve
        list of cutoffs and the associated TPR, FNR, accuracy
        # why: I may want to pick up the cutoff for which 200*TPR + 1000*TNR is minimum
"""

#@@ to package - RV will do - but you need to read it

#%% Test / explanation
import random
random.seed(123)

p1 = [random.uniform(0,1) for i in range(10)]
p1.sort(reverse=True)
o1 = ["Y","Y","N","N","Y","Y","N","N","N","N"]
rocd = pd.DataFrame({'pred':p1, 'outcome':o1})

totalpos = len([i for i in o1 if i == "Y" ])
totalneg = len(o1) - totalpos

rocd['Yes'] = 0
rocd['No'] = 0
rocd['Sens'] = 0
rocd['Spec'] = 0

#%%
for i in range(rocd.shape[0]):
    tcut = rocd.loc[i,'pred']
    tcond = rocd.pred > tcut
    tposreal = rocd.outcome == 'Y'
    tp = rocd.loc[(tcond) & (tposreal)].shape[0]
    fp = rocd.loc[(tcond) & (~tposreal)].shape[0]
    tn = rocd.loc[(~tcond)&(~tposreal)].shape[0]
    fn = rocd.loc[(~tcond) & (tposreal)].shape[0]
    rocd.loc[i,'Yes'] = tp
    rocd.loc[i,'No'] = tn
    rocd.loc[i,'sens'] = tp/totalpos
    rocd.loc[i,'spec'] = tn/totalneg

rocd['oppspec'] = 1-rocd['spec']
plt.plot(rocd['oppspec'], rocd['sens'])
#%%
# ROC curve
import sklearn.metrics as skm
skm.roc_curve(y, QDAz)
#=================
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 2 # binary classification

for i in range(n_classes):
    fpr[i], tpr[i], _ = skm.roc_curve(y==i, QDAzp[:,i])
    roc_auc[i] = skm.roc_auc_score(y==i, QDAzp[:,i])
# technically we don't need both curves, one will do...
    
# further reading: https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# see references in there for ROC vs Precision-Recall curve "when to" arguments

#Plot of a ROC curve for a specific class

#%%
           
plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
