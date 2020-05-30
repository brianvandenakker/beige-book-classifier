# -*- coding: utf-8 -*-
"""
Purpose: Identify 'best' Random Forest models for interest rate increase/decrease prediction.
Inputs: Train and Test data
Outputs: A list of locally optimal AUC and Accuracy scores for prediction performance on observed results: Did we correctly predict interest rate descisions?
"""

#%% Random Forest
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier as RFClass
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#%%
model_rf = RFClass()
print(model_rf.get_params())

#Specify potential parameters to apply
#Assisted by: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

#1 Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300, num =6)]
#2 Number of features to consider at every split
max_features = ['auto']
#3 Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 7, num = 5)]
#4 Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
#5 Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
#6 Create the random grid
random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

#%%

#Model Selection
t0=DT.datetime.now()
results = [-1, 0, 1]
modellist = []

#Purpose: Identifies 'best' parameters for each RF to predict result through a randomized search via cross validation
#Output: A list of the best local models given the specified search space. 
#Note: This trains CV = 3 * n_iter = 50 * digits = 10 = 1,500 random forests. 
for result in results:
    t0=DT.datetime.now()
    #Split train data 50/50
    resultIdentifier(train, result, train = True)
    #Use the random grid to search for best parameters
    # Search of parameters, using 3 fold cross validation with 50 random combinations
    rf_random = RandomizedSearchCV(estimator = model_rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=2019)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    #Identify strongest parameters by accuracy
    rf_random.best_params_
    #Apply strongest parameters to the rf model
    model_rf = RFClass(n_estimators = rf_random.best_params_['n_estimators'], 
                       min_samples_split =  rf_random.best_params_['min_samples_split'], 
                       min_samples_leaf = rf_random.best_params_['min_samples_leaf'], 
                       max_features = 'auto', 
                       max_depth = rf_random.best_params_['max_depth'])
    #Append best model for each digit (0-9) to list
    modellist.append(model_rf)
    
t1=DT.datetime.now()
print('Randomized Search of Random Forest for possible results took ' + str(t1-t0))
#Randomized Search of Random Forest for possible results took 0:00:33.480838'
#Ran w/0 (n_jobs = -1)
#%%
#Performance

#Purpose: Find scoring values to compare with other models.
#Output: A list of AUC and Accuracy scores for each digit.

AUC = []
accuracy = []
#Specify a cost for false positive and false negatives. Choose 50/50 here. No obvious reason to lean more or less 'conservative'
fpc = 100
fnc = 100

t0=DT.datetime.now()
for result in results:
    #Call train and test sets
    resultIdentifier(train, result, train = True)
    resultIdentifier(test, result, train = False)
    #Fit the model selected with the locally optimal parameters
    modellist[result].fit(X_train, y_train)
    #Use the RF to predict on the test data
    predict = modellist[result].predict_proba(X_test)[:,1]
    #Compute false positive rate, true positive rate, and the subsequent threshold
    fpr_rf, tpr_rf, thresh_rf = skm.roc_curve(y_test, predict)
    #Append AUC to list
    AUC.append(skm.auc(fpr_rf,tpr_rf))
    #Compute costs for false positives and negatives for shifting thresholds
    totalcost = fpc*fpr_rf + [fnc*(1-x) for x in tpr_rf]
    #Locate minimum total cost 
    minpos = np.argmin(totalcost)
    #Locate the cutoff value
    cutoff = thresh_rf[minpos]
    #Apply the cutoff to determine T/F
    predict = [True if x > cutoff else False for x in predict]
    #Isolate 'accuracy' and append to the accuracy measure list for comparison to other prediction methods.
    accuracy.append(confusionMatrixInfo(predict, y_test)['accuracy'])
    print(confusionMatrixInfo(predict, y_test))
    
t1 = DT.datetime.now() 
print('Computing performance measures took '+ str(t1-t0))
print(accuracy)

#Accuracy
#[0.6666666666666666, 0.5555555555555556, 0.6944444444444444]
#%%

#Look at feature importances. Must specify the digit in modellist[i]
feature_importances = pd.DataFrame(modellist[9].feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance'])\
                                   .sort_values('importance',
                                                ascending=False)

#Plot top twenty important features
plt.plot(feature_importances[:20])

#Plot one tree
from graphviz import Source
from sklearn import tree as treemodule

treePlot = Source(treemodule.export_graphviz(
        modellist[1].estimators_[1]
        , out_file=None
        , feature_names=X_train.columns
        , filled = True
        , proportion = True
        )
)

#%%