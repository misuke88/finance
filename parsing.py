from collections import Counter
import csv
import gzip
import re
import time

from gensim import corpora, models, similarities
from nltk.corpus import stopwords

from settings import DATA_DIR, DIR_8K, DIR_PRICE
from utils import checkdir, file_read, get_today, get_version


KEYS = 'FILE TIME EVENTS TEXT ITEM'.split()


def get_id_docs_from_gz(company_code, error_filename):

    def get_id_doc_price(doc, error_filename):
        lines = filter(None, doc.split('\n'))
        id_ = lines[0].split('/')[-1].split('.')[0]
        doc = ' '.join([line for line in lines\
                if not (any(line.startswith(k) for k in KEYS) or line=='</DOCUMENT>')])
        price = get_close_price_from_price_history(company_code, id_, error_filename)
        return (id_, doc, price)

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

    with gzip.open('%s/%s.gz' % (DIR_8K, company_code)) as f:
        docs = filter(None, f.read().split("<DOCUMENT>"))

    return [get_id_doc_price(d, error_filename) for d in docs]


def parse_doc(doc):
    # TODO: remove special characters
    doc = doc.lower()
    doc = doc.replace('\t', ' ').replace('\n', ' ')
    doc = re.sub('\s+', ' ', doc)
    return doc


def append_id_docs_to_file(id_docs_price, filename):
    with open(filename, 'a') as f:
        for i in id_docs_price:
            id_, doc, price = i[0], parse_doc(i[1]), i[2]
            # f.write('%s\t%.4f\t%s\n' % (id_,float(price), doc))
            f.write('%s\t%s\t%.4f\n' % (id_, doc, float(price)))


if __name__ == '__main__':

    company_code = 'C'
    filename = '%s/stock.tsv' % DATA_DIR
    error_filename = '%s/errorfilename.txt' %DATA_DIR
    open(error_filename, 'w').close()
    open(filename, 'w').close()     # clear file
    id_docs_price = get_id_docs_from_gz(company_code, error_filename)
    append_id_docs_to_file(id_docs_price, filename)
    print('%s\t%s' % (company_code, len(id_docs_price)))
