import csv
import numpy as np
import matplotlib.pyplot as plt


from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from settings import DATA_DIR

def openfiles(filename):
	doc_list = []
	responses = []
	''' make list with all contents'''
	with open(filename, 'rb') as f:
		docs = f.readlines()
	open(filename).close()	
	for doc in docs:
		doc = doc.split('\t')
		doc_list.append(doc[1])
		responses.append(doc[2])
	return doc_list, responses

def tokenizing(docs):
	vectorizer = CountVectorizer(lowercase = False,  min_df = 5)
	matrix_td = vectorizer.fit_transform(docs) # term doc matrix
	return matrix_td

def logistic_regression(X_train, X_test, y_train, y_test):
	logreg = linear_model.LogisticRegression(C=1e5)
	model = logreg.fit(X_train, y_train)
	predicted = model.predict(X_test)
	probs = model.predict_proba(X_test)
	accuracy = metrics.accuracy_score(y_test, predicted)
	confusion_matrix = metrics.confusion_matrix(y_test, predicted)
	report = metrics.classification_report(y_test, predicted)
	return accuracy

if __name__ == '__main__':

	filename = '%s/stock_ratio.txt' % DATA_DIR
	docs, y = openfiles(filename)
	X = tokenizing(docs) # term doc matrix
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	logit_accuracy = logistic_regression(X_train, X_test, y_train, y_test)
	scores = cross_val_score(linear_model.LogisticRegression(), X, y, scoring ='accuracy', cv = 10 )
	print scores, scores.mean()
