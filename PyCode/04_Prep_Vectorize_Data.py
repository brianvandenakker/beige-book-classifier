# -*- coding: utf-8 -*-
#Method 2: Vectorize the text with term frequency - inverse document frequency (tfidf) 

#%%
from sklearn.utils import shuffle

df = pd.DataFrame(data)
#Train/test split
train, test = skl_traintest_split(df[['text', 'sentiment', 'y']].copy(), test_size = 0.20, random_state = 42)
X_test = test['text']
y_test = test['y']

#Must train 3 models: when rates rise (or not), dont' change(or do), fall (or not)
#Ex. Focus on Rates rise or not --> return 'y' value of 1 if rise, else 0.
def classify_data(value = None):
    if None:
        raise Exception("You must select some value.")
    if value == -1:
        train_dec = train.copy()
        for index, val in train_dec.iterrows():
            if val['y'] != value:
                train_dec.loc[index, 'y'] = 0
            else:
                train_dec.loc[index, 'y'] = 1
        #Shuffle data to enable better learning
        train_dec = shuffle(train_dec)
        #Find number of 'true' values that met the conditional
        true_val = sum(train_dec['y'])
        if true_val > len(train_dec)/2:
            true_val = int(round(len(train_dec)/2))
        #Limit the number of 'false' values to equal the number of true values
        #This balances the training data. The idea is the learning algorithms will be able
        #to find more nuance in the data if the data is balanced.
        balanced_data_in = train_dec[train_dec['y'] == 1].head(true_val)
        balanced_data_out = train_dec[train_dec['y'] == 0].head(true_val)
        train_dec = shuffle(balanced_data_in.append(balanced_data_out))
        return train_dec
    elif value == 0:
        train_neut = train.copy()
        for index, val in train_neut.iterrows():
            if val['y'] != value:
                train_neut.loc[index, 'y'] = 0
            else:
                train_neut.loc[index, 'y'] = 1
        train_neut = shuffle(train_neut)
        #Find number of 'true' values that met the conditional
        true_val = sum(train_neut['y'])
        if true_val > len(train_neut)/2:
            true_val = int(round(len(train_neut)/2))
        #Limit the number of 'false' values to equal the number of true values
        #This balances the training data. The idea is the learning algorithms will be able
        #to find more nuance in the data if the data is balanced.
        balanced_data_in = train_neut[train_neut['y'] == 1].head(true_val)
        balanced_data_out = train_neut[train_neut['y'] == 0].head(true_val)
        train_neut = shuffle(balanced_data_in.append(balanced_data_out))
        return train_neut
    else:
        train_inc = train.copy()
        for index, val in train_inc.iterrows():
            if val['y'] != value:
                train_inc.loc[index, 'y'] = 0
            else:
                pass
        train_inc = shuffle(train_inc)
        #Find number of 'true' values that met the conditional
        true_val = sum(train_inc['y'])
        if true_val > len(train_inc)/2:
            true_val = int(round(len(train_inc)/2))
        #Limit the number of 'false' values to equal the number of true values
        #This balances the training data. The idea is the learning algorithms will be able
        #to find more nuance in the data if the data is balanced.
        balanced_data_in = train_inc[train_inc['y'] == 1].head(true_val)
        balanced_data_out = train_inc[train_inc['y'] == 0].head(true_val)
        train_inc = shuffle(balanced_data_in.append(balanced_data_out))
        return train_inc
    
    
# Because of the format of the *y vals, we have to adjust them into booleans. 
def adjust_test_format(value = None):
    if None:
        raise Exception("You must select some value.")
    y_vals = []
    if value == -1:
        for y in y_test:
            if y == -1:
                y_vals.append(1)
            else: y_vals.append(0)
        return y_vals
    elif value == 0:
        for y in y_test:
            if y == 0:
                y_vals.append(1)
            else: y_vals.append(0)
        return y_vals
    elif value == 1:
        for y in y_test:
            if y == 1:
                y_vals.append(1)
            else: y_vals.append(0)
        return y_vals
    
    
 
#Lemmatize text and remove stopwords along with words of length == 1
def tokenizer(text):
    return [word.lemma_ for word in nlp(text) if not (word.is_stop or len(word)==1)]

#Produces n tfidf vectorizations as defined by *iterations. 
#max_df, min_df, ngram_range are lists of paramaters to be randomly searched. 
def vectorizer(tokenizer, iterations, max_df_options, min_df_options, ngram_range_options):
    start = time.time()
    print("Vectorizing Training Text...")
    #A complete search would be computationally heavy. Random search requires less computation. 
    max_df = random.choice(max_df_options)
    min_df = random.choice(min_df_options)
    ngram_range = random.choice(ngram_range_options)
    #Define vectorizor w/ random parameters
    tfidf = TfidfVectorizer(tokenizer = tokenizer, max_df=max_df, min_df=min_df, ngram_range=ngram_range)
    #Fit tfidf to the text data in our corpus and append to list of vectors. ngram_range is a list of tuples where for example (1,2) would apply a bigram search. 
    vec = tfidf.fit_transform(X_train)
    #Generate dictionaries of vector information
    vec_list.append({'vec_iter' : iterations, 'vec_id': len(vec_list)+1, 'vec':vec, 
                         'tfidf':tfidf, 'vocab_length':len(tfidf.vocabulary_), 
                         'max_df':max_df, 'min_df': min_df, 'ngram_range':ngram_range})   
    print(f"Compute time for vectorizor() of {iterations} iterations: {time.time() - start} seconds")
    return vec_list


#Ex. 
#iterations = 1
#max_df = [0.95, 0.9, 0.85, 0.8]
#min_df = [2,3,4,5]
#ngram_range = [(1,1), (1,2),(1,3)]
  
        
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        