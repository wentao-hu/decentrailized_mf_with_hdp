'''
Modified on Oct 12, 2021
1.modify the dataprocessing step for explicit feedback
'''

import numpy as np


class Dataset_explicit(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix, self.num_users, self.num_items ,self.train_num_rated_users= self.load_explicit_rating_file_as_list(
            path + ".train.rating")
        self.testRatings, self.num_test_users, self.num_test_items, self.test_num_rated_users= self.load_explicit_rating_file_as_list(
            path + ".test.rating")

    def load_explicit_rating_file_as_list(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        # Get the number of rated users for each item 
        num_rated_users={}
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                ratingList.append([user, item, rating])
                if item in num_rated_users.keys():
                    num_rated_users[item]+=1
                else:
                    num_rated_users[item]=1
                num_users = max(num_users, user)
                num_items = max(num_items, item)
                line = f.readline()
        return ratingList, num_users+1, num_items+1,num_rated_users
