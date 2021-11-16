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
        self.trainMatrix, self.user_dict, self.item_dict,self.train_num_rated_users= self.load_explicit_rating_file_as_list(
            path + ".train.rating")
        self.testRatings, _, _, self.test_num_rated_users= self.load_explicit_rating_file_as_list(
            path + ".test.rating")

    def load_explicit_rating_file_as_list(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        user_dict,item_dict={},{}
        user_index,item_index=0,0
        # Get the number of rated users for each item in original rating matrix
        num_rated_users={}
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                ratingList.append([user, item, rating])
                
                #mark users and items with index start from 1
                if user not in user_dict.keys():
                    user_dict[user]=user_index
                    user_index+=1
                if item not in item_dict.keys():
                    item_dict[item]=item_index
                    item_index+=1

                
                #count the number of rating users for each item
                if item in num_rated_users.keys():
                    num_rated_users[item]+=1
                else:
                    num_rated_users[item]=1
                # num_users = max(num_users, user)
                # num_items = max(num_items, item)
                line = f.readline()
        user_dict={v: k for k,v in user_dict.items()}
        item_dict={v: k for k,v in item_dict.items()}
        return ratingList, user_dict, item_dict, num_rated_users
