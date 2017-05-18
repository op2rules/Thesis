import random
import nltk
import numpy as np
import copy

#Scikit-Learn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline #simplification
from sklearn.linear_model import SGDClassifier #linear svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV #optimization

#NLTK

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

class StemmerTokenizer(object):
    def __init__(self):
        self.lcs = PorterStemmer()
    def __call__(self, doc):
        return [self.lcs.stem(t) for t in word_tokenize(doc)]

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

execfile('parse.py') # get the data (5477 ISEAR and 4090 AMAN)
sentences = isearSentences + amanSentences #+ semevalSentences
emotions = isearEmotions + amanEmotions #+ semevalEmotions

# scikit vectorizers expect regular sentences
for i in range(len(sentences)):
  sentences[i] = ' '.join(sentences[i])


#shuffle the data and preserve correct label
combined = zip(sentences,emotions)
random.shuffle(combined)
sentences, emotions = zip(*combined)
emotions = list(emotions) # they come out as a tuple
sentences = list(sentences)

testing_size = 4500
testing_size2 = testing_size/3 # subset for the combiner to test/train on

# KB Classifier
execfile('kbclassifier.py')
kbproba = kbclassifier(sentences[:testing_size], synMode = 0) # probability vectors
kbpred = copy.copy(kbproba)
for i in range(len(kbpred)):
    kbpred[i] = EVConverter(kbpred[i])
kbpred = np.asarray(kbpred) # emotion predictions
print('KB: %s' %(np.mean(kbpred == emotions[:testing_size])))
#print(metrics.classification_report(emotions[:testing_size],kbpred))

# MNB
# 5 run - average : 66.5% accuracy on all the data
from sklearn.preprocessing import StandardScaler

mnb_clf = Pipeline([('vect', CountVectorizer(tokenizer=StemmerTokenizer())), 
		     ('clf', MultinomialNB())])
mnb_clf = mnb_clf.fit(sentences[testing_size:], emotions[testing_size:])
mnb = mnb_clf.predict(sentences[:testing_size])
mnbproba = mnb_clf.predict_proba(sentences[:testing_size])
print('MNB: %s' %(np.mean(mnb == emotions[:testing_size])))
#print(metrics.classification_report(emotions[:testing_size],mnb))


# SVM 5 run - average : 65.5% accuracy on all the data
# Logistic regression used to provide more data to the combiner
svm_clf = Pipeline([('vect', CountVectorizer(tokenizer=StemmerTokenizer())),
                     ('clf', SGDClassifier(loss='log', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = svm_clf.fit(sentences[testing_size:], emotions[testing_size:])
svm = svm_clf.predict(sentences[:testing_size])
svmproba = svm_clf.predict_proba(sentences[:testing_size])
print('SVM: %s' %(np.mean(svm == emotions[:testing_size])))
#print(metrics.classification_report(emotions[:testing_size],svm))

# ME 6 run - average : 69.4% accuracy on all the data
me_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer())),
		     ('clf', LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None))])
mefit = me_clf.fit(sentences[testing_size:], emotions[testing_size:])
me = me_clf.predict(sentences[:testing_size])
meproba = me_clf.predict_proba(sentences[:testing_size])
print('ME: %s' %(np.mean(me == emotions[:testing_size])))
#print(metrics.classification_report(emotions[:testing_size],me))

# Majority Vote Combiner

majority = np.ndarray(testing_size,'S2')
for i in range(testing_size):
    if mnb[i] == svm[i]:
        majority[i] = svm[i]
    else:
        majority[i] = me[i]

print('Majority Vote: %s' %(np.mean(majority == emotions[:testing_size])))

# Conversion of individual classifier output into 21-tuple

predictionVector = np.ndarray(shape=(testing_size,28))  
predictionVector2 = np.ndarray(shape=(testing_size,21)) # No KB classifier
for i in range(len(mnbproba)):
    predictionVector[i][:7] = mnbproba[i]
    predictionVector[i][7:14] = svmproba[i]
    predictionVector[i][14:21] = meproba[i]
    predictionVector2[i] = copy.copy(predictionVector[i][:21])
    predictionVector[i][21:28] = kbproba[i]

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(7),random_state=1)
mlp.fit(predictionVector[testing_size2:], emotions[testing_size2:testing_size])

nn = mlp.predict(predictionVector[:testing_size2])
print('MLP: %s' %(np.mean(nn == emotions[:testing_size2])))
#print(metrics.classification_report(emotions[:testing_size2],nn))

mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(7),random_state=1)
mlp2.fit(predictionVector2[testing_size2:], emotions[testing_size2:testing_size])
nn = mlp2.predict(predictionVector2[:testing_size2])
print('MLP (no KB): %s' %(np.mean(nn == emotions[:testing_size2])))
#print(metrics.classification_report(emotions[:testing_size2],nn))

competitor = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
_ = competitor.fit(predictionVector[testing_size2:], emotions[testing_size2:testing_size])
svm2 = competitor.predict(predictionVector[:testing_size2])
print('SVM: %s' %(np.mean(svm2 == emotions[:testing_size2])))

competitor2 = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
_ = competitor2.fit(predictionVector2[testing_size2:], emotions[testing_size2:testing_size])
svm3 = competitor2.predict(predictionVector2[:testing_size2])
print('SVM (no KB): %s' %(np.mean(svm3 == emotions[:testing_size2])))

# Print faults
"""
for i in range(100):
    if nn[i] != emotions[i]:
        print nn[i]
        print sentences[i]
        print emotions[i]
"""