from collections import Counter
import csv
import gzip
import re
import time

import nltk

from settings import DATA_DIR, DIR_8K, DIR_PRICE
import utils


KEYS = 'FILE TIME EVENTS TEXT ITEM'.split()
TOTAL_INDEX = 'djia gspc ixic vix'.split() # dow jones. snp500, nasdaq, vol

def get_id_docs_from_gz(company_code, error_filename, error_filename_total_index):

    def get_id_doc_price(doc, error_filename):
        total = []
        lines = filter(None, doc.split('\n'))
        id_ = lines[0].split('/')[-1].split('.')[0]
        doc = ' '.join([line for line in lines\
                if not (any(line.startswith(k) for k in KEYS) or line=='</DOCUMENT>')])
        price = get_close_price_from_price_history(company_code, id_, error_filename)
        for INDEX in range(len(TOTAL_INDEX)):
            total.append(get_close_index_from_total_index(TOTAL_INDEX[INDEX], id_, error_filename_total_index))
        return (id_, doc, price, total)

    def get_close_price_from_price_history(company_code, id_, error_filename):

        with open('%s/%s.csv' % (DIR_PRICE, company_code)) as csvfile:
            historys = list(csv.reader(csvfile, delimiter= ','))

        datestring = id_.split('-')[2][0:8]
        date = time.strptime(datestring, '%Y%m%d')
        date = time.strftime('%Y-%m-%d', date)
        price = 0
        for history in historys:
            if history[0]==date:
                price = history[6]
        if price == 0:
            price = '0'
            with open(error_filename, 'a') as ef:
                ef.write('%s\n' % id_)
        return price

    def get_close_index_from_total_index(use_index, id_, error_filename_total_index):

        with open('%s/%s.csv' % (DIR_PRICE, use_index)) as csvfile:
            historys = list(csv.reader(csvfile, delimiter= ','))

        datestring = id_.split('-')[2][0:8]
        date = time.strptime(datestring, '%Y%m%d')
        date = time.strftime('%Y-%m-%d', date)
        price = 0
        for history in historys:
            if history[0]==date:
                price = history[6]
        if price == 0:
            price = '0'
            with open(error_filename_total_index, 'a') as ef:
                ef.write('%s\t%s\n' % (use_index, id_))
        return price

    with gzip.open('%s/%s.gz' % (DIR_8K, company_code)) as f:
        docs = filter(None, f.read().split("<DOCUMENT>"))

    return [get_id_doc_price(d, error_filename) for d in docs]


def parse_doc(doc):
    # TODO: remove special characters
    # TODO: remove stopwords?

    SUBSTITUTIONS = [
        ('\.?\d+(,\d+)*(\.\d+)?', ' NUM '),                 # numbers
        (r'(\W)\1{3,}', r'\1\1\1'),                         # repetitive symbols
    ]

    lowercase = doc.lower()
    flattened = lowercase.replace('\t', ' ').replace('\n', ' ')
    converted = utils.re_sub(flattened, SUBSTITUTIONS)  # replace special tokens

    return ' '.join(nltk.tokenize.wordpunct_tokenize(converted))


def append_id_docs_to_file(id_docs_price, filename):
    with open(filename, 'a') as f:
        for i in id_docs_price:
            id_, doc, price = i[0], parse_doc(i[1]), i[2]
            dow, snp, nas, vol = i[3][0], i[3][1], i[3][2], i[3][3]
            f.write('%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % \
                    (id_, doc, float(price), float(dow), float(snp), float(nas), float(vol)))


if __name__ == '__main__':

    company_codes = 'C WFC GS JPM BAC'.split()
    filename = '%s/stock.txt' % DATA_DIR
    error_filename = '%s/errorfilename.txt' %DATA_DIR
    error_filename_total_index = '%s/errorfilename_total_index.txt' %DATA_DIR
    num_total_doc = 0
    open(error_filename, 'w').close()
    open(filename, 'w').close()     # clear file
    open(error_filename_total_index, 'w').close()
    for company_code in company_codes:
        id_docs_price = get_id_docs_from_gz(company_code, error_filename, error_filename_total_index)
        append_id_docs_to_file(id_docs_price, filename)
        print('%s\t%s' % (company_code, len(id_docs_price)))
        num_total_doc = num_total_doc + len(id_docs_price)

    print('total\t%d' % num_total_doc)
