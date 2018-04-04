#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:24:05 2018

@author: cheng-xili
"""


import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
import help_fun as hf
import os
import sklearn
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import zscore
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc

import csv
import codecs
import numpy as np
import pandas as pd
import os
import re


ENCODING = 'utf-8'
path = "./"

read_filename_1 = 'train.csv'
read_filename_2 = 'test.csv'
write_filename = 'merged.csv'


with codecs.open(read_filename_1, "r", ENCODING) as infile_1, \
    codecs.open(read_filename_2, "r", ENCODING) as infile_2, \
     codecs.open(write_filename, "w", ENCODING) as outfile:
     
    reader_1 = csv.reader(infile_1)
    reader_2 = csv.reader(infile_2)
    headers = next(reader_1) # headers
    writer = csv.writer(outfile)    
    writer.writerow(headers)
    
    for row in reader_1:
        writer.writerow(row)
        
    next(reader_2)
    for row in reader_2:
        writer.writerow(row)
        


path = ""
filename_read = os.path.join(path,"merged.csv")
df = pd.read_csv(filename_read,na_values=['NA','?','nan', ' '])
#df_tmp = df.drop('SalePrice', axis=1, inplace=False)
#df_tmp = df.drop('Id', axis=1, inplace=False)


for column in list(df.columns.values): 
    if column == 'SalePrice':
        continue
    if df[column].dtype == object:
        hf.encode_text_dummy(df, column)
    else:
        hf.missing_median(df, column)

df_train_x = df.iloc[0:1460,]
df_train_y = df[['SalePrice']]
df_train_y = df_train_y.iloc[0:1460,]
df_test_x = df.iloc[1460:,]

df_test_x.drop('SalePrice', axis=1, inplace=True)
df_train_x.drop('SalePrice', axis=1, inplace=True)

for column in list(df_test_x.columns.values): 
    hf.missing_median(df_test_x, column)



forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(df_train_x, df_train_y)






y_test_pred = forest.predict(df_test_x)
y_test_pred = pd.DataFrame(y_test_pred)

output_fileName = 'y_test.csv'

y_test_pred['Id'] = y_test_pred.index + 1461
y_test_pred.rename(columns={0: 'SalePrice'}, inplace=True)
y_test_pred = y_test_pred[['Id', 'SalePrice']]
y_test_pred.to_csv(output_fileName,index=True)  






