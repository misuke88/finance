import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from settings import DATA_DIR, LOG_DIR
import utils

TOTAL_INDEX = 'djia gspc ixic vix'.split() # dow jones. snp500, nasdaq, vol
EXPID = utils.get_expid()
utils.set_logger('%s/%s.log' % (LOG_DIR, EXPID), 'DEBUG')


def openfiles(filename, arg):

	data = pd.read_csv(filename, sep='\t', header = 0)
	if arg == 100:
		value = data['text']
	else:
		value = data[TOTAL_INDEX[arg]]

	return value


def preprocessing(docs, y, arg):

	code = TOTAL_INDEX[arg]
	idx = y[y != 'ERROR'].index.tolist()
	docs = docs.loc[idx]
	y = y.loc[idx]
	return list(docs), list(y)


def tokenizing(docs):

	vectorizer = CountVectorizer(lowercase = False,  min_df = 5)
	matrix_td = vectorizer.fit_transform(docs) # term doc matrix
	return matrix_td


def generate_LR(X_train, X_test, y_train, y_test):

	logreg = linear_model.LogisticRegression(C=1e5)
	model = logreg.fit(X_train, y_train)
	predicted = model.predict(X_test)
	probs = model.predict_proba(X_test)
	accuracy = metrics.accuracy_score(y_test, predicted)
	cm = metrics.confusion_matrix(y_test, predicted)
	report = metrics.classification_report(y_test, predicted)
	return cm, accuracy


def generate_RF(X_train, X_test, y_train, y_test):

    rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=3)
    rf.fit(X_train.toarray(), y_train)
    y_pred = rf.predict(X_test.toarray())
    cm = confusion_matrix(y_test, y_pred)
    return cm, rf.score(X_test.toarray(), y_test)


def cross_validation_10(X, y):

	lr_scores = cross_val_score(linear_model.LogisticRegression(), X, y, scoring ='accuracy', cv = 10 )
	rf_scores = cross_val_score(RandomForestClassifier(), X.toarray(), y, scoring ='accuracy', cv = 10 )
	print "CV: Accuracy of Logistic Regression is %.2f\n, and Accuracy of Random Forest is %.2f\n."\
								 % (lr_scores.mean, rf_scores.mean)

if __name__ == '__main__':

	filenameX = '%s/stock_X.txt' % DATA_DIR
	filenameY = '%s/stock_Y.txt' % DATA_DIR
	docs= openfiles(filenameX, 100)
	y = openfiles(filenameY, 1) # arg = 1: SNP500
	docs, y = preprocessing(docs, y, arg=1)
	X = tokenizing(docs) # term doc matrix
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	print "Modeling of logistic regression..."
	lr_cm, lr_accuracy = generate_LR(X_train, X_test, y_train, y_test) #logistic regression
 	print "Modeling of random forest..."
	rf_cm, rf_accuracy = generate_RF(X_train, X_test, y_train, y_test) # #random forest

	# print lr_accuracy, rf_accuracy, lr_cm, rf_cm
	print "Accuracy of Logistic Regression is %.2f\n, and Accuracy of Random Forest is %.2f\n."\
								 % (lr_accuracy, rf_accuracy)

	print "confusion matrix of logistic regression \n", lr_cm, \
				"\nconfusion_matrix of random forest\n", rf_cm
