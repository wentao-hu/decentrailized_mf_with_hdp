'''
author:Wentao Hu
'''
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
def load_rating_file_as_list(filename):        
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


def main():
    ratingList = []
    with open("Data/ml-1m/ratings.dat", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("::")
            user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
            ratingList.append([user, item, rating])             
            line = f.readline() 

    #generate cross validation datasets for ml-1m
    df=pd.DataFrame(ratingList)
    kf=KFold(n_splits=5,shuffle=True,random_state=1)
    file_index=1
    for train_index,test_index in kf.split(df):
        train,test=df.iloc[train_index],df.iloc[test_index]
        train.to_csv(f"Data/ml-1m/u{file_index}.base",sep=' ', index=False, header=False)
        test.to_csv(f"Data/ml-1m/u{file_index}.test",sep=' ', index=False, header=False)
        file_index+=1

    #generate train and test dataset for ml-1m   
    df=df.rename(columns={0:"user",1:"item",2:"rating"})
    test=df.groupby("user").sample(n=10,random_state=1)
    train=df.drop(test.index)
    test.to_csv("Data/ml-1m/u.test",sep=' ', index=False, header=False)
    train.to_csv("Data/ml-1m/u.base",sep=' ', index=False, header=False)


if __name__=="__main__":
    main()
