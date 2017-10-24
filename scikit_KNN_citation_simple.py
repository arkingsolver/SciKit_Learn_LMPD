# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:27:21 2017

@author: Admin
"""

import pandas as pd

############### READ IN DATA SOURCE ####################
### READ DIFFRENT SECTIONS FOR TRAIN, TEST
df_train = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv', nrows=100000, skipinitialspace=True)
headers = list(df_train) # Save Headers fot dfs that skip 1st row.

df_test = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv',names=headers,skiprows=100001, nrows=5000, skipinitialspace=True)

#Drop NaN
df_train =df_train.dropna(axis=0, how='any')
df_test =df_test.dropna(axis=0, how='any')

#Convert object d-types to category d-types
obj_columns = df_train.select_dtypes(['object']).columns
df_train[obj_columns] = df_train[obj_columns].apply(lambda x: x.astype('category'))
df_test[obj_columns] = df_test[obj_columns].apply(lambda x: x.astype('category'))

#Convert category d-types to number values that can be used in array.
cat_columns = df_train.select_dtypes(['category']).columns
df_train[cat_columns] = df_train[cat_columns].apply(lambda x: x.cat.codes)
df_test[cat_columns] = df_test[cat_columns].apply(lambda x: x.cat.codes)


############ CREATE DATASETS & TARGETS ################
df_train_data = df_train.drop('ACTIVITY_RESULTS', 1)
train_data =  df_train_data.values
train_target = df_train.ACTIVITY_RESULTS.values

df_test_data = df_test.drop('ACTIVITY_RESULTS', 1)
test_data =  df_test_data.values
test_target = df_test.ACTIVITY_RESULTS.values

############ CREATE & APPLY MODEL #####################
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=50, p=2,
           weights='uniform')

# Fit model to training data
model.fit(train_data, train_target) 

# Use fitted model to predit classification
predict = model.predict(test_data)

############### OUTPUT RESULTS OF PREDICTION ##########
#load prediction values and real target values from test into df
df_output = pd.DataFrame({'Prediciton': predict,'Target': test_target})

#Compare values to find if they match
df_output['Match'] = df_output['Prediciton'] == df_output['Target']

#Get % accuracy scores
accuracy = df_output.Match.value_counts(normalize=True)

print (accuracy)