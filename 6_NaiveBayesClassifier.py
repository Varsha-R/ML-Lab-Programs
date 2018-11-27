'''
Assuming a set of documents that need to be classified, use the naïve Bayesian
Classifier model to perform this task.

'''
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data["target_names"])

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle='true')
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle='true')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

classify = Pipeline([('vect',TfidfVectorizer()),('clf',MultinomialNB())])
classify.fit(news_train.data, news_train.target)
predicted = classify.predict(news_test.data)

from sklearn import metrics as sm
from sklearn.metrics import accuracy_score, confusion_matrix

test_accuracy = accuracy_score(news_test.target, predicted)
test_confusion = confusion_matrix(news_test.target, predicted)
clf_report = sm.classification_report(news_test.target, predicted, target_names=news_test.target_names)
print("Accuracy: ", test_accuracy) 
print("Confusion matrix: \n", test_confusion)
print(clf_report)
