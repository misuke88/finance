from collections import Counter
import csv
import gzip
import re
import time
import datetime
import pandas as pd

import nltk
from dateutil.relativedelta import relativedelta

from settings import DATA_DIR, DIR_8K, DIR_PRICE, DIR_SNP
import utils

from utils import checkdir, file_read, re_sub, get_version, get_datetime

KEYS = 'FILE TIME EVENTS TEXT ITEM'.split()
TOTAL_INDEX = 'djia gspc ixic vix'.split() # dow jones. snp500, nasdaq, vol

def get_id_docs_from_gz(company_code, error_filename, error_filename_total_index):

    def get_id_doc_price(doc, error_filename):
        total = []
        lines = filter(None, doc.split('\n'))
        id_ = next(lines).split('/')[-1].split('.')[0]
        doc = ' '.join([line for line in lines\
                if not (any(line.startswith(k) for k in KEYS) or line=='</DOCUMENT>')])
        price, week_move, month_move, quater_move, year_move = get_close_price_from_price_history(company_code, id_, error_filename)
        for INDEX in range(len(TOTAL_INDEX)):
            total.append(get_close_index_from_total_index(TOTAL_INDEX[INDEX], id_, error_filename_total_index))
        return (id_, doc, price, week_move, month_move, quater_move, year_move, total)

    def get_close_price_from_price_history(company_code, id_, error_filename):

        def make_numeric_input_variable(historys, h,id_, now_price):

            def get_movement(historys, h, id_, now_price, arg, prev):
                
                now = get_datetime(id_)
                for p in range(h, min(h+6+arg*25, len(historys))):
                    t = historys[p][0]
                    if t <= prev:
                        return (float(now_price) - float(historys[p][6]))/float(historys[p][6])
                return 0

            week_move, month_move, quater_move, year_move =[],[],[],[]
            now_date = get_datetime(id_)
            now_date = datetime.datetime.strptime(now_date, '%Y-%m-%d')

            week_move = get_movement(historys, h, id_, now_price,0, \
                datetime.date.isoformat(now_date - datetime.timedelta(weeks=1)))
            month_move = get_movement(historys, h, id_, now_price, 1, \
                datetime.date.isoformat(now_date - relativedelta(months=+1)))
            quater_move =get_movement(historys, h, id_, now_price, 3, \
                datetime.date.isoformat(now_date - relativedelta(months=+3)))
            year_move = get_movement(historys, h, id_, now_price, 12, \
                datetime.date.isoformat(now_date - relativedelta(years=+1)))

            return week_move, month_move, quater_move, year_move

        with open('%s/%s.csv' % (DIR_PRICE, company_code)) as csvfile:
            historys = list(csv.reader(csvfile, delimiter= ','))

        date = get_datetime(id_)
        price = 0
        for h in range(len(historys)):
            if historys[h][0]==date:
                price = historys[h][6]
                week_move, month_move, quater_move, year_move = make_numeric_input_variable(historys,h, id_, price)
        
        if price == 0:
            price, week_move, month_move, quater_move, year_move = '0', 0,0,0,0
            with open(error_filename, 'a') as ef:
                ef.write('%s\n' % id_)

        return price, week_move, month_move, quater_move, year_move

    def get_close_index_from_total_index(use_index, id_, error_filename_total_index):

        with open('%s/%s.csv' % (DIR_PRICE, use_index)) as csvfile:
            historys = list(csv.reader(csvfile, delimiter= ','))

        date = get_datetime(id_)
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
        docs = filter(None, f.read().decode('utf-8').split("<DOCUMENT>"))

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
            week, month, quater, year = i[3], i[4], i[5], i[6]
            dow, snp, nas, vol = i[7][0], i[7][1], i[7][2], i[7][3]
            f.write('%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n' % \
                    (id_, doc, float(price), week, month, quater, year,\
                     float(dow), float(snp), float(nas), float(vol)))

def get_company_list(sector):

    def openfiles(filename):

        data = pd.read_csv(filename, sep='\t', header =None)
        data.columns = ['company', 'abb', 'sector']
        return data

    def stats(num_sectors):
        
        stat = dict()
        for i in num_sectors:
            stat[i] = len(companys[companys['sector']==i]['sector'])
            # print(i, stat[i])
        return stats

    filename = '%s/snp1500_20120928.txt' % DIR_SNP
    companys = openfiles(filename)
    num_sectors = set(companys['sector'])
    finance_list = companys[companys['sector']=='Financials']['abb']
 
    return finance_list


if __name__ == '__main__':

    # company_codes = 'C WFC GS JPM BAC USB AXP SPG AIG MET'.split()
    company_codes = get_company_list('Financials')
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
