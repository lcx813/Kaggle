# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import re
import csv
import codecs

ENCODING = 'utf-8'
LOW = 0
HIGH = 999999

read_filename = 'train.csv'

with codecs.open(read_filename, "r", ENCODING) as infile:

    reader = csv.reader(infile)
    next(reader) # headers

    word_set = set()
    for row in reader: 
        words = row[1].split(" ")
        for item in words:
            word_set.add(item)