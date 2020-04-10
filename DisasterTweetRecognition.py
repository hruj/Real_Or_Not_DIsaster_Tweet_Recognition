#Disaster Tweets Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import random
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#importing data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

#Creating x and y axes
x = train_data['text']
y = train_data['target']

#Removing stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

#On training data
for i in range(0,7613):
    tweet = x[i]
    tweet = tweet.lower()
    tweet = tweet.split() 
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
        
#Creating a Bag of Words model using corpus
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
tweet = cv.fit_transform(corpus).toarray()

#Splitting training data into training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(tweet,y,test_size=0.2,random_state = 123)
 
#Using classification model to classify useful words......
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier()
classifierRF.fit(x_train,y_train)

#Predicting tweets
y_predRF = classifierRF.predict(x_test)

#Preparing confusion matrix to calculate model score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cmRF = confusion_matrix(y_test,y_predRF)
accuracy_score(y_test,y_predRF)

tweets_test = test_data['text']

#Creating a Bag of Words model using corpus
test_x = cv.transform(tweets_test)
#Now using model on test set
y_pred = classifierRF.predict(test_x)

#Creating sample submission
submission['target'] = y_pred
submission.to_csv("submission.csv")
submission.head()
