from Dataset_explicit import Dataset_explicit
import numpy as np

import os
import csv
import argparse

dataset = Dataset_explicit('Data/ml-1m')
train_rating_matrix= dataset.trainMatrix
test_ratings = dataset.testRatings
train_num_rated_users=dataset.train_num_rated_users
user_dict=dataset.user_dict
item_dict=dataset.item_dict

num_users=dataset.num_users
num_items=dataset.num_items
print(1805 in item_dict.values())

