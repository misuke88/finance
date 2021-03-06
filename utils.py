#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from datetime import datetime
import json
import logging
import os
import re
import time


get_expid = lambda: datetime.today().isoformat().replace(':', '-')


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def file_read(filename):
    with gzip.open(filename) as f:
        # d = f.read().replace('.', '')
        # doc = d.split("<DOCUMENT>")

        d = f.read().split("<DOCUMENT>")
    return d

def get_datetime(id_):
    
    datestring = id_.split('-')[2][0:8]
    date = time.strptime(datestring, '%Y%m%d')
    date = time.strftime('%Y-%m-%d', date)
    return date

def get_version():
    with open('version.cfg', 'r') as f:
        return f.read().strip()


def re_sub(text, substitutions):
    for sub in substitutions:
        text = re.sub(sub[0], sub[1], text)
    return text

def set_logger(filename, level='DEBUG'):
    # basic config
    log_formatter = logging.Formatter('%(asctime)-19.19s %(levelname)s > %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.handlers: # remove previous loggers
        root_logger.handlers = []

    # log to file
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger
