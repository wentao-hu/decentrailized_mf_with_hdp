import argparse
from random import triangular
from dataprocess import *
import numpy as np
import random
import csv
import logging 
from sklearn.model_selection import KFold
from utils_private import *
from mf_sampling_decentralized import *
from mf_hdp_decentralized import *
import pandas as pd

data="Data/ml-100k"
i=2
train_data=load_rating_file_as_list(f"{data}/u.data")
user_dict,item_dict=get_user_and_item_dict(train_data) 
num_users=max(max(user_dict.values()),len(user_dict))
num_items=max(max(item_dict.values()),len(item_dict))
print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #validation_pairs=%d' %
    (num_users, num_items, len(train_data),len([1,2,3])))

df=pd.DataFrame(train_data)
kf=KFold(n_splits=5,shuffle=True,random_state=1)
for train_index,test_index in kf.split(df):
    train,test=df.iloc[train_index],df.iloc[test_index]
    print(type(train),type(test))
    print(test.iloc[:10])
