import csv
import numpy as np
import matplotlib.pyplot as plt


from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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

def generate_LR(X_train, X_test, y_train, y_test):
	logreg = linear_model.LogisticRegression(C=1e5)
	model = logreg.fit(X_train, y_train)
	predicted = model.predict(X_test)
	probs = model.predict_proba(X_test)
	accuracy = metrics.accuracy_score(y_test, predicted)
	confusion_matrix = metrics.confusion_matrix(y_test, predicted)
	report = metrics.classification_report(y_test, predicted)
	return accuracy

def generate_RF(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=3)
    rf.fit(X_train.toarray(), y_train)
    return rf.score(X_test.toarray(), y_test)

if __name__ == '__main__':

	filename = '%s/stock_ratio.txt' % DATA_DIR
	docs, y = openfiles(filename)
	X = tokenizing(docs) # term doc matrix
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
	
	#logistic regression
	logit_accuracy = generate_LR(X_train, X_test, y_train, y_test)
	# #random forest
	rfs = [generate_RF(X_train, X_test, y_train, y_test) for i in xrange(10)]
	rf_accuracy = np.mean(rfs)
	print logit_accuracy, rf_accuracy
	# # print "Accuracy of Logistic Regression is %.2f\%\n, and Accuracy of Random Forest is %.2f\%\n." % (logit_accuracy, rf_accuracy)

	#cross validataion method
	lr_scores = cross_val_score(linear_model.LogisticRegression(), X, y, scoring ='accuracy', cv = 10 )
	rf_scores = cross_val_score(RandomForestClassifier(), X.toarray(), y, scoring ='accuracy', cv = 10 )
	# print "CV: Accuracy of Logistic Regression is %.2f\%\n, and Accuracy of Random Forest is %.2f\%\n." % (lr_scores.mean*100, rf_scores.mean*100)
	print lr_scores.mean(), rf_scores.mean()