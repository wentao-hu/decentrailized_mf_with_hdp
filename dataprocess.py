'''
author:Wentao Hu
'''
import numpy as np
import os
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(0)



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
    with open("Data/ml-100k/u.data", "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
            ratingList.append([user, item, rating])             
            line = f.readline()     

    #generate different sparsity dataset for ml-1m (0.8,0.6,0.4,0.2 fraction version)
    for fraction in [0.2,0.4,0.6,0.8,1]:
        df=pd.DataFrame(ratingList)
        df=df.rename(columns={0:"user",1:"item",2:"rating"})
        dir=f"Data/ml-100k-{fraction}"
        if not os.path.exists(dir):
            os.makedirs(dir)

        df=df.groupby(df["user"]).apply(lambda x:x.sample(frac=fraction))  #sampling groupby by user

        kf=KFold(n_splits=5,shuffle=True,random_state=1)
        file_index=1
        for train_index,test_index in kf.split(df):
            train,test=df.iloc[train_index],df.iloc[test_index]
            train.to_csv(f"{dir}/u{file_index}.base",sep='\t', index=False, header=False)
            test.to_csv(f"{dir}/u{file_index}.test",sep='\t', index=False, header=False)
            file_index+=1

        #using leave one out strategy to generate train and test dataset
        df=df.rename(columns={"user":"user2"}) 
        df=df.reset_index() 
        df=df[["user2","item","rating"]]
        test=df.groupby("user2").sample(n=1,random_state=1)
        train=df.drop(test.index)
        test.to_csv(f"{dir}/u.test",sep='\t', index=False, header=False)
        train.to_csv(f"{dir}/u.base",sep='\t', index=False, header=False)


if __name__=="__main__":
    main()