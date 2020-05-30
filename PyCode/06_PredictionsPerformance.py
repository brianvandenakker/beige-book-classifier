"""
Final Predictions from "Best" Model
"""
#Final Model: Logistic Regression

#Make Global Predictions
y_train = train['y']
X_train= train.drop(columns = {'y', 'book_id'})
X_train = X_train.fillna(X_train.mode().iloc[0])

y_test = test['y']
X_test = test.drop(columns = {'y', 'book_id'})
X_test = X_test.fillna(X_train.mode().iloc[0])                          

#Generate dictionary of final probabalistic predictions
   

#Set to dataframe
predictions = pd.DataFrame(prob_pred).T
predictions = predictions.rename(index=int, columns={0: -1, 1:0, 2:1})
#Select value with highest predictive confidence and store in predValue
predValue = predictions.idxmax(axis = 1)

#Write predictions to output folder
predictFile = os.path.join(outputFld, "decision_predictions")
predValue.to_pickle(predictFile + ".pkl")
predValue.to_csv(predictFile+".csv", index = None)

#Show the confusion matrix
cfmatrix = confusion_matrix(y_test, predValue)

#Calculate the global accuracy
tptn = 0
result = [-1, 0, 1]
for i in result:
    tptn += cfmatrix[i,i]
accuracy = tptn/len(y_test)

#Call classification report
target_names = ['decrease_rates', 'no_change', 'increase_rates']
report = classification_report(y_test, predValue, target_names = target_names)

#%%
#Summary of Results:

print('Global Confusion Matrix: ' + '\n'  + str(cfmatrix) + '\n')
#Global Confusion Matrix: 
#[[ 1  5  1]
# [ 1 18  4]
# [ 0  5  1]]

print('Global Accuracy: ' + str(accuracy) + '\n')
#Global Accuracy: 0.5555555555555556

print('Classification Report: ' + '\n' + str(report))
#Classification Report: 
#                precision    recall  f1-score   support
#
#decrease_rates       0.50      0.14      0.22         7
#     no_change       0.64      0.78      0.71        23
#increase_rates       0.17      0.17      0.17         6
#
#     micro avg       0.56      0.56      0.56        36
#     macro avg       0.44      0.36      0.36        36
#  weighted avg       0.54      0.56      0.52        36
