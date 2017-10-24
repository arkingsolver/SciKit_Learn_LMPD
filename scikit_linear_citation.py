# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:27:21 2017

@author: Admin
"""

import pandas as pd

### READ IN DATA SOURCE
### READ DIFFRENT SECTIONS FOR TRAIN, TEST, PREDICT
df_train = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv', nrows=100000, skipinitialspace=True)
# SAve Headers for the next selections that skip header row.
headers = list(df_train)

df_test = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv',names=headers,skiprows=100001, nrows=1000, skipinitialspace=True)
#df_predict = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv',names=headers,skiprows=93001, nrows=100, skipinitialspace=True)

df_train =df_train.dropna(axis=0, how='any')
df_test =df_test.dropna(axis=0, how='any')
#df_predict =df_predict.dropna(axis=0, how='any')


obj_columns = df_train.select_dtypes(['object']).columns
df_train[obj_columns] = df_train[obj_columns].apply(lambda x: x.astype('category'))
df_test[obj_columns] = df_test[obj_columns].apply(lambda x: x.astype('category'))
#df_predict[obj_columns] = df_predict[obj_columns].apply(lambda x: x.astype('category'))


cat_columns = df_train.select_dtypes(['category']).columns
df_train[cat_columns] = df_train[cat_columns].apply(lambda x: x.cat.codes)
df_test[cat_columns] = df_test[cat_columns].apply(lambda x: x.cat.codes)
#df_predict[cat_columns] = df_predict[cat_columns].apply(lambda x: x.cat.codes)
#####################################################

########################
df_train_data = df_train.drop('ACTIVITY_RESULTS', 1)
train_data =  df_train_data.values
train_target = df_train.ACTIVITY_RESULTS.values
##############
df_test_data = df_test.drop('ACTIVITY_RESULTS', 1)
test_data =  df_test_data.values
test_target = df_test.ACTIVITY_RESULTS.values
############################


# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
model = linear_model.LinearRegression()

model.fit(train_data, train_target) 

predict = model.predict(test_data)

df_output = pd.DataFrame(
    {'Prediciton': predict,
     'Target': test_target
    })


df_output.Prediciton = df_output.Prediciton.round()

df_output['Match'] = df_output['Prediciton'] == df_output['Target']

accuracy = df_output.Match.value_counts(normalize=True)

print (accuracy)