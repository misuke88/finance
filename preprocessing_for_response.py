import csv
import pandas as pd
import numpy as np
import datetime

from dateutil.relativedelta import relativedelta
from settings import DATA_DIR
from pandas import DataFrame

from utils import checkdir, file_read, re_sub, get_version, get_datetime

TOTAL_INDEX = 'djia gspc ixic vix'.split() # dow jones. snp500, nasdaq, vol


def openfiles(filename):

    data = pd.read_csv(filename, sep='\t', header =None)
    data.columns = ['id', 'text', 'closePrice', 'week', 'month', 'quater', 'year','djia', 'gspc', 'ixic', 'vix']
    return data


def split_X_Y_dataset(data, error_filename_ratio):

    columns = ['week', 'month', 'quater', 'year', 'vix','text']
    ids  = data['id']
    dataX = pd.DataFrame(index =ids[range(len(ids))], columns = columns)
    dataX['week'] = np.asarray(data['week'])
    dataX['month'] = np.asarray(data['month'])
    dataX['quater'] = np.asarray(data['quater'])
    dataX['year'] = np.asarray(data['year'])
    dataX['vix'] = np.asarray(data['vix'])
    dataX['text'] = np.asarray(data['text'])

    dataX = dataX.drop(dataX.index[len(dataX)-1])
    dataY = change_price_of_number_of_days(data, error_filename_ratio)

    return dataX, dataY


def change_price_of_number_of_days(data, error_filename_ratio):
## compute price change between n-1's closed price and n+7's closed price
## n: news event

    columns = TOTAL_INDEX
    idxs =[]
    ndays = 7
    error = 0
    criteria = 0.01
    prices = data['closePrice']
    ids = data['id']
    tmp_ids = ids[1:len(ids)-ndays-1]
    change_price = pd.DataFrame(index =tmp_ids, columns=columns)

    for INDEX in range(len(TOTAL_INDEX)-1):
        tmp_change =[]
        total = data[TOTAL_INDEX[INDEX]]
        print (INDEX)
        for i in range(1, len(prices)-ndays-1):
            # the price of previous day = null  then logging error
            if prices[i+ndays] == 0 or total[i+ndays] == 0:
                with open(error_filename_ratio, 'a') as ef: # log that document
                    ef.write('%s\t%s\n' % (TOTAL_INDEX[INDEX], ids[i]))
                tmp_change.append('ERROR')
                error += 1
            else:
                ratio = (prices[i+ndays]-prices[i-1])/prices[i+ndays]
                total_ratio = (total[i+ndays]-total[i-1])/total[i+ndays]
                if ratio-total_ratio >= criteria:
                    tmp_change.append('UP')
                elif ratio-total_ratio <= -criteria:
                    tmp_change.append('DOWN')
                else:
                    tmp_change.append('STAY')
        change_price[TOTAL_INDEX[INDEX]] = np.asarray(tmp_change)
    change_price[TOTAL_INDEX[INDEX+1]] = np.asarray(data[TOTAL_INDEX[INDEX+1]][1:len(data)-ndays-1])

    print ('The %d days have null prices.' % error)
    return change_price


def write_table(data, filename):
    data.to_csv(filename, sep='\t', header=True)


if __name__ == '__main__':

    filename = '%s/stock_4company.txt' % DATA_DIR
    filename_X = '%s/stock_4company_X_7days.txt' % DATA_DIR
    filename_Y = '%s/stock_4company_Y_7days.txt' % DATA_DIR
    error_filename_ratio = '%s/errorfilename_ratio.txt' % DATA_DIR

    open(error_filename_ratio, 'w').close()
    open(filename_X, 'w').close()
    open(filename_Y, 'w').close()

    data = openfiles(filename)
    dataX, dataY = split_X_Y_dataset(data, error_filename_ratio)
    write_table(dataX, filename_X)
    write_table(dataY, filename_Y)
