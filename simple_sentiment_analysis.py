"""
===================================================
Simple sentiment analysis for the english language
===================================================

one-class-binary-classifier (1 : good / 0 : bad) 

Sentiment analysis based on Support Vector Machine (SVM) by scikit-learn with data provided by Michigan State University and the IMDB through Kaggle.
Contains automatic bag-of-words creation from training set with simple stopword removal.

Source:
 * data:        http://inclass.kaggle.com/c/si650winter11
 * stopwords:   http://www.ranks.nl/stopwords

External files:
 * training.txt - contains all labeld training data, with columns (label,text)
 * stopwords.txt - contains all stopwords with one word per line
 
Visit "https://github.com/libeanim/simple-sentiment-analysis" for further information.
"""

import numpy as np
import pandas as pd
import re
import os
from sklearn import svm

__version__ = '1.0'

print(__doc__)
print('version:', __version__)

print('Prepare data...')
# getting data
path = os.path.dirname(os.path.realpath(__file__)) + '/'
train = pd.read_csv(path + "training.txt")
train['text'] = train['text'].map(lambda x: re.sub('[^A-Za-z ]+', '', x).lower())
stopwords = pd.read_csv(path + "stopwords.txt")

# split into single words
out = []
for text in train['text']:
    word_list = text.split(' ')
    out = word_list + out
    
data = pd.DataFrame(data=out)

# drop all (english) stopwords
data = data.drop(data[data[0] == ""].index)
for word in stopwords:
    data = data.drop(data[data[0] == word].index)
bow = data[0].value_counts()[0:500].index

# shuffle data
train = train.reindex(index=np.random.permutation(train.index), columns=['label', 'text'])
Xtemp = np.array([train['text'].str.contains(s, na=False).as_matrix() for s in bow]).T
Ytemp = train['label'].as_matrix()

# set point for splitting data in training set and cross validation set
cutpoint = int(Xtemp.shape[0] * 4.0/5.0)

# training set
Xtrain = Xtemp[0:cutpoint]
Ytrain = Ytemp[0:cutpoint]

# cross validation set
Xcv = Xtemp[cutpoint:]
Ycv = Ytemp[cutpoint:]
print('done.')

# train SVM from sklearn
print('Train SVM...')
clf = svm.SVC()
clf.fit(Xtrain, Ytrain)
print('done.')

# check accuracy - DEPRECATED use built in function instead
print('Training set: ' + str(100 * np.sum(clf.predict(Xtrain) == Ytrain)/Xtrain.shape[0]) + '% accuracy')
print('Cross validation set: ' + str(100 * np.sum(clf.predict(Xcv) == Ycv)/Xcv.shape[0]) + '% accuracy')
