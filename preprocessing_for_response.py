import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

from pprint import pprint

from sklearn.feature_extraction.text import CountVectorizer
from settings import DATA_DIR


def openfiles(filename, error_filename_ratio):
	doc_list = []
	price_list = []
	id_list =[]
	''' make list with all contents'''
	with open(filename, 'rb') as f:
		docs = f.readlines()
	open(filename).close()	
	for doc in docs:
		doc = doc.split('\t')
		id_list.append(doc[0])
		doc_list.append(doc[1])
		price_list.append(float(doc[2]))
	responses = change_price_against_previous_day(id_list,doc_list, price_list, error_filename_ratio)
	doc_list[-1]=[] # adjust the number of rows same as responses
	price_list[-1]=[]
	return id_list, doc_list, responses

def change_price_against_previous_day(ids,docs, prices, error_filename_ratio):
	# print prices[1], len(prices)
	change_price = []
	error = 0
	criteria = 0.01
	for i in range(len(prices)-1):
		if prices[i+1] == 0: # the price of previous day = null  then logging error
			with open(error_filename_ratio, 'a') as ef: # log that document
				ef.write('%s\n' % ids[i+1])			
			error += 1
			ids[i] =[] #delete row
			docs[i] =[]
		else: 
			ratio = (prices[i+1]-prices[i])/prices[i+1]
			if ratio >= criteria:
				change_price.append('UP')
			elif ratio <= -criteria:
				change_price.append('DOWN')
			else:
				change_price.append('STAY')
	print 'The %d days have null prices.' % error
	return change_price

def append_id_docs_ratio_to_file(ids, docs, responses, filename):
    with open(filename, 'a') as f:
        for i in range(len(responses)):
            f.write('%s\t%s\t%s\n' % (ids[i], docs[i], responses[i]))


if __name__ == '__main__':

	filename = '%s/stock.txt' % DATA_DIR
	filename_ratio = '%s/stock_ratio.csv' % DATA_DIR
	error_filename_ratio = '%s/errorfilename_ratio.txt' % DATA_DIR
	open(error_filename_ratio, 'w').close()
	open(filename_ratio, 'w').close()
	ids, docs, responses = openfiles(filename, error_filename_ratio)
	append_id_docs_ratio_to_file(ids, docs, responses, filename_ratio)

	