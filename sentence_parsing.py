import pandas as pd
import numpy as np

import nltk

from settings import DATA_DIR
from pandas import DataFrame

from settings import DATA_DIR, DIR_8K, DIR_PRICE
import utils

from utils import checkdir, file_read, re_sub, get_version, get_datetime

TOTAL_INDEX = 'djia gspc ixic vix'.split() # dow jones. snp500, nasdaq, vol

def openfiles(filename, arg):

    data = pd.read_csv(filename, sep='\t', header = 0)
    data = data.where((pd.notnull(data)), '')   # Replace np.nan with ''
    if arg == 100: # X
        value = pd.DataFrame(data)
        value.index = data['id']
    else: # y
        columns = TOTAL_INDEX[arg]
        value = pd.DataFrame(data[TOTAL_INDEX[arg]])
        value.index = data['id']
    return value

def preprocessing(docs, y, arg):

    code = TOTAL_INDEX[arg]
    idx = y[y[code] != 'ERROR'].index.tolist()
    X = docs.loc[idx]
    y = y.loc[idx]
    idx = y[y[code] != 'STAY'].index.tolist()
    X = docs.loc[idx]
    y = y.loc[idx]
    print len(y)
     
    return X, y

def parse_sentence(X, y):

    ids = X['id']
    columns =['text', 'sentiment']
    data_senti = pd.DataFrame(index =ids[range(len(ids))], columns = columns)
    sentences =[]
    senti =[]
    for id_ in ids:
        value = y.loc[id_]
        if value.item() =='UP':
            senti.append(1)
        elif value.item() =='DOWN':
            senti.append(-1)
        else:
            senti.append(0)
        sentences.append('\n'.join(nltk.sent_tokenize(X.loc[id_, 'text'])))

    # data_senti['ids'] = np.asarray(X['id'])

    data_senti['text'] = np.asarray(sentences)
    data_senti['sentiment'] =np.asarray(senti)
   
    return data_senti

def gather_data(X):

    print 'start gather_data'

    stock_pos = '\n'.join(X[X['sentiment']==1]['text'])
    stock_neg = '\n'.join(X[X['sentiment']==-1]['text'])

    return stock_pos, stock_neg


if __name__ == '__main__':

    filenameX = '%s/stock_4company_X_7days.txt' % DATA_DIR
    filenameY = '%s/stock_4company_Y_7days.txt' % DATA_DIR


    X = openfiles(filenameX, 100)
    y = openfiles(filenameY, 1) # arg = 1: SNP500
    X, y = preprocessing(X, y, arg=1)

    data_senti = parse_sentence(X, y)
    output_pos = '%s/stock_4company_7days.pos' % DATA_DIR
    output_neg = '%s/stock_4company_7days.neg' % DATA_DIR
    stock_pos, stock_neg = gather_data(data_senti)

    print len(stock_pos), len(stock_neg)

    with open(output_pos, 'w') as f:            # clear file
        f.write('%s' % stock_pos)
    with open(output_neg, 'w') as f:
        f.write('%s' % stock_neg)
