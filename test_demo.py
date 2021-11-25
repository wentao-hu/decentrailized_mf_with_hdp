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

dataset = Dataset_explicit("Data/ml-1m")
train_ratings = dataset.trainMatrix



user_dict,item_dict=get_user_and_item_dict(train_ratings)
num_users=max(max(user_dict.values()),len(user_dict))
num_items=max(max(item_dict.values()),len(item_dict))


user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,[0.1,0.2,1],[0.54,0.37,0.09],[0.1,0.2,1])

max_budget=1
ratingList=train_ratings
sum_privacy=0
min_privavy=9999
max_privacy=-9999
for i in range(len(ratingList)):
    user,item=ratingList[i][0],ratingList[i][1]
    user_privacy_weight=user_privacy_vector[user]
    item_privacy_weight=item_privacy_vector[item]
    rating_privacy_budget = user_privacy_weight*item_privacy_weight * max_budget
    sum_privacy+=rating_privacy_budget
    min_privacy=min(min_privacy,rating_privacy_budget)
    max_privacy=max(max_privacy,rating_privacy_budget)


threshold=sum_privacy/len(ratingList)
