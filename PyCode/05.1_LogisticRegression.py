# -*- coding: utf-8 -*-
"""
Logistic Regression: Predictions for the Fed's Interest Rate Decisions
"""

#%% Logistic regression
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics as skm

#PARAMETERS!
iterations = 4
max_df = [0.95, 0.9, 0.85, 0.8]
min_df = [2,3,4,5]
ngram_range = [(1,1), (1,2),(1,3)]
model_performance = []
clss = [-1, 0, 1]
vec_list = []
for cl in clss:
    #Constrain text to 50/50 true false split 
    df = classify_data(cl)
    X_train = df['text']
    y_train = df['y']
    #Begin random search of vectorizor parameters for tfidf. 
    for iteration in range(iterations):
        vec_list = vectorizer(tokenizer=tokenizer,iterations=iterations, max_df_options=max_df, min_df_options=min_df, ngram_range_options=ngram_range)   
        X_train = vec_list[iteration]['vec']
        print("Vecorization complete, now training the model")
        # Fit initial model
        logit = LogReg(random_state = 42, solver='lbfgs', max_iter=1000).fit(X_train,y_train)
        #Select optimal features
        model = SelectFromModel(logit, prefit=True)
        X_new = model.transform(X_train)
        #Fit new model to selected features
        logit_new = LogReg(random_state = 42, solver='lbfgs', max_iter=1000).fit(X_new,y_train)
        #Transform test text according to fitted tfidf from training set
        print("Vectorizing Test Data")
        test_vec = vec_list[iteration]['tfidf'].transform(X_test)
        test_vec_new = model.transform(test_vec)
        #Make predictions from the model with the test data
        print("Calculating predictions")
        prediction = logit_new.predict(test_vec_new)
        fpr_rf, tpr_rf, thresh_rf = skm.roc_curve(list(adjust_test_format(cl)), list(prediction))
        #Append AUC to list
        auc = skm.auc(fpr_rf,tpr_rf)
        #Specify a cost for false positive and false negatives. Choose 50/50 here. No obvious reason to lean more or less 'conservative'
        fpc = 100
        fnc = 100
        #Compute costs for false positives and negatives for shifting thresholds
        totalcost = fpc*fpr_rf + [fnc*(1-x) for x in tpr_rf]
        #Locate minimum total cost 
        minpos = np.argmin(totalcost)
        #Locate the cutoff value
        cutoff = thresh_rf[minpos]
        #Apply the cutoff to determine T/F
        predict = logit_new.predict_proba(test_vec_new)
        predict = [True if x > cutoff else False for x in [prediction[0] for prediction in predict]]
        #Print confusion matrix to evaluate performance
        accuracy = confusionMatrixInfo(predict,adjust_test_format(cl))['accuracy']
        no_information_rate = confusionMatrixInfo(predict,adjust_test_format(cl))['no information rate']
        print(f"Complete! Model Performance: {accuracy - no_information_rate}")
        model_performance.append({'performance': (accuracy-no_information_rate),'classifier':cl, 'iteration': iteration, 'model': logit_new})












