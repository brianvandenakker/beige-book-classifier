# -*- coding: utf-8 -*-
"""
Neural Network
"""

#%%
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report, confusion_matrix 

#%%

sentimentData = pd.read_csv(os.path.join(savedDataFld, "sentiment_data.csv"))
train, test = skl_traintest_split(sentimentData.copy(), test_size = 0.20, random_state = 2019)

y_train = train['y']
X_train= train.drop(columns = {'y', 'book_id'})
X_train = X_train.fillna(X_train.mode().iloc[0])

y_test = test['y']
X_test = test.drop(columns = {'y', 'book_id'})
X_test = X_test.fillna(X_train.mode().iloc[0])
        
#%% Initial model: No parameter tuning
mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=1000)  
mlp.fit(X_train, y_train) 


X_test = test.drop(columns={'y', 'book_id'})
y_test = test['y']
X_test = X_test.fillna(X_train.mode().iloc[0])
predictions = mlp.predict(X_test)  

 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions)) 
confusionMatrixInfo(predictions, y_test)['accuracy']

#{'confusionMatrix': array([[ 2,  4,  1],
#                           [ 1, 17,  5],
#                           [ 0,  5,  1]], dtype=int64),
# 'accuracy': 0.5277777777777778,
# 'no information rate': 0.8333333333333334,
# 'sensitivity': 0.3333333333333333,
# 'specificity': 0.9444444444444444}
#%%

parameter_space = {
    'hidden_layer_sizes': [(30,30,30), (50,100,50), (100, 100, 100), (10, 10, 10)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],}

#Apply RandomizedSearchCV for parameter tuning with 3-fold CV

rs = RandomizedSearchCV(mlp, parameter_space, cv=3)

rs.fit(X_train, y_train)

print('Best local parameters found:\n', rs.best_params_)
#Best local parameters found:
# {'solver': 'adam', 'learning_rate': 'adaptive', 'hidden_layer_sizes': (50, 100, 50), 
#  'alpha': 0.05, 'activation': 'logistic'}

predictions = rs.predict(X_test)

confusionMatrixInfo(predictions, y_test)

#{'confusionMatrix': array([[ 0,  7,  0],
#                           [ 0, 23,  0],
#                           [ 0,  6,  0]], dtype=int64),
# 'accuracy': 0.6388888888888888,
# 'no information rate': 0.8055555555555556,
# 'sensitivity': 0.0,
# 'specificity': 1.0}

