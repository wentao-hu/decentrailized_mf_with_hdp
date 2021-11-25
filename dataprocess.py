'''
author:Wentao Hu
'''

import numpy as np


class Dataset_explicit(object):
    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix= self.load_rating_file_as_list(
            path + ".train.rating")
        self.testRatings= self.load_rating_file_as_list(
            path + ".test.rating")

    def load_rating_file_as_list(self, filename):        
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                ratingList.append([user, item, rating])             
                line = f.readline()   
        return ratingList


def get_user_and_item_dict(ratingList):
    user_dict,item_dict={},{}
    user_index,item_index=0,0
    for i in range(len(ratingList)):
        (user, item, rating) = ratingList[i]
        #mark users and items with index start from 0
        if user not in user_dict.keys():
            user_dict[user]=user_index
            user_index+=1
        if item not in item_dict.keys():
            item_dict[item]=item_index
            item_index+=1
    user_dict={v: k for k,v in user_dict.items()}
    item_dict={v: k for k,v in item_dict.items()}
    return user_dict,item_dict


def get_num_rated_user(ratingList):
    # Get the number of rated users for each item
    num_rated_users={}
    for i in range(len(ratingList)):
        (user, item, rating) = ratingList[i]
        if item in num_rated_users.keys():
            num_rated_users[item]+=1
        else:
            num_rated_users[item]=1
    return num_rated_users

def string_to_list(str):
    '''transfer a numerical string to list, used in parse some arguments below'''
    tmp=str.split(" ")
    lst=[float(x) for x in tmp]
    return lst