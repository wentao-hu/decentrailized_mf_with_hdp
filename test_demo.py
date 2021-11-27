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

data="Data/ml-100k"
i=2
train_data=load_rating_file_as_list(f"{data}/u{i}.base")
user_dict,item_dict=get_user_and_item_dict(train_data) 
num_users=max(max(user_dict.values()),len(user_dict))
num_items=max(max(item_dict.values()),len(item_dict))
print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #validation_pairs=%d' %
    (num_users, num_items, len(train_data),len([1,2,3])))

#get privacy vector and stretched rating
user_privacy_list=[0.2,0.6,1]
user_type_fraction=[0.54,0.37,0.09]
item_privacy_list=[0.2,0.6,1]

user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_fraction,item_privacy_list)

stretch_ratings=stretch_rating(train_data,user_privacy_vector,item_privacy_vector)
print(type(stretch_ratings))

# user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,[0.1,0.2,1],[0.54,0.37,0.09],[0.1,0.2,1])

# max_budget=1
# ratingList=train_ratings
# sum_privacy=0
# min_privavy=9999
# max_privacy=-9999
# for i in range(len(ratingList)):
#     user,item=ratingList[i][0],ratingList[i][1]
#     user_privacy_weight=user_privacy_vector[user]
#     item_privacy_weight=item_privacy_vector[item]
#     rating_privacy_budget = user_privacy_weight*item_privacy_weight * max_budget
#     sum_privacy+=rating_privacy_budget
#     min_privacy=min(min_privacy,rating_privacy_budget)
#     max_privacy=max(max_privacy,rating_privacy_budget)

# threshold=sum_privacy/len(ratingList)
