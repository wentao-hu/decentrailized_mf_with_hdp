import argparse
from random import triangular
from dataprocess import Dataset_explicit
import numpy as np
import random
import csv
import logging 


dataset = Dataset_explicit("Data/ml-1m")
train_rating_matrix = dataset.trainMatrix
test_ratings = dataset.testRatings
user_dict=dataset.user_dict
item_dict=dataset.item_dict

train_users=user_dict.values()
train_items=item_dict.values()

for i in range(len(test_ratings)):
    (user, item, rating) = test_ratings[i]
    #only consider users and items appear in training dataset
    if user not in train_users:
        print("user not in",test_ratings[i])
    if item not in train_items:
        print("item not in",test_ratings[i])