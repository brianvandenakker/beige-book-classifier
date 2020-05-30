# -*- coding: utf-8 -*-

#Method 1: Count positive, negative, and uncertain words from a predefined list.




#Import Dictionary Classifiers as lists
negative = (pd.read_csv(os.path.join(rawDataFld, "negative.csv"), header = None))[0].tolist()
positive = (pd.read_csv(os.path.join(rawDataFld, "positive.csv"), header = None))[0].tolist()
uncertain = (pd.read_csv(os.path.join(rawDataFld, "uncertain.csv"), header = None))[0].tolist()

negative = [word.lower() for word in negative]
positive = [word.lower() for word in positive]
uncertain = [word.lower() for word in uncertain]


#Import text
#with open(raw_text, 'r') as f:
#    json.loads(raw_text, f)



nlp = spacy.load('en_core_web_sm')

#Count negative, positive, and uncertain words within each book.
for index, book in enumerate(data):
    neg = 0
    unc = 0
    pos = 0
    for word in nlp(data[index]['text'].lower()):
        if str(word) in negative:
            neg +=1
        elif str(word) in uncertain:
            unc +=1
        elif str(word) in positive:
            pos +=1
        else:
            pass
    tot = neg + unc + pos
    #Save counts as a percentage of total words found in 'classifier lists'
    data[index]['sentiment'] = {'positive':pos/tot, 'negative':neg/tot, 'uncertain':unc/tot}
        
#%%
#Add interest rate decision as dependent variable. 1 = rate increase, 0 = no change, -1 = rate decrease
rate_decision = pd.read_csv(os.path.join(rawDataFld, "interest_rate_decision.csv"), header = None)[0].tolist()

for meeting, val in enumerate(rate_decision):
    data[meeting]['y'] = val


#Write data to json
sentiment_data = os.path.join(savedDataFld, "sentiment_data.json").replace('\\', '/')
with open(sentiment_data, 'w') as f:
    json.dump(data, f)
