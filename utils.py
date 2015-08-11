#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

from datetime import datetime
import json
import os

get_today = lambda : datetime.now()


def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def file_read(filename):
    with gzip.open(filename) as f:
        # d = f.read().replace('.', '')
        # doc = d.split("<DOCUMENT>")

        d = f.read().split("<DOCUMENT>")
    return d


def get_version():
    with open('version.cfg', 'r') as f:
        return f.read().strip()
