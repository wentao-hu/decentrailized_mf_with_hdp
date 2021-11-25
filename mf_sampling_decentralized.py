# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
@author: Wentao Hu (stevenhwt@gmail.com)
'''

import argparse
from random import triangular
from dataprocess import *
import numpy as np
import random
import csv
import logging
from sklearn.model_selection import KFold
from utils_private import *



def evaluate(model, test_ratings,user_privacy_vector,item_privacy_vector):
    '''Evaluate loss on the test dataset,sampling method does not need to scale back '''
    absolute_loss = 0.0
    square_loss=0
    num_test_examples=0
    for i in range(len(test_ratings)):
        (user, item, rating) = test_ratings[i]
        #only consider users and items appear in training dataset
        if user in user_privacy_vector.keys() and item in item_privacy_vector.keys():
            num_test_examples+=1
            prediction = model._predict_one(user, item)
            if prediction>=5:
                prediction=5
            if prediction<=1:
                predicion=1
            err_ui = rating - prediction
            absolute_loss += abs(err_ui)
            square_loss+=err_ui**2

    mae=absolute_loss/num_test_examples
    mse=square_loss/num_test_examples
    return mae,mse


def get_threshold(ratingList,user_privacy_vector,item_privacy_vector,max_budget,strategy):
    #get the sampling threshold according to sampling stategy
    sum_budget=0
    min_budget=9999
    max_budget=-9999
    for i in range(len(ratingList)):
        user,item=ratingList[i][0],ratingList[i][1]
        user_privacy_weight=user_privacy_vector[user]
        item_privacy_weight=item_privacy_vector[item]
        rating_privacy_budget = user_privacy_weight*item_privacy_weight * max_budget
        sum_budget+=rating_privacy_budget
        min_budget=min(min_budget,rating_privacy_budget)
        max_budget=max(max_budget,rating_privacy_budget)
    
    if strategy=="mean":
        threshold=sum_budget/len(ratingList)
    elif strategy=="min":
        threshold=min_budget
    else:
        threshold=max_budget
    return threshold


def get_sampled_ratings(ratingList,threshold,max_budget,user_privacy_vector,item_privacy_vector):
    sampled_index = []
    # Sampling the rating with heterogeneous probability
    for i in range(len(ratingList)):
        user,item=ratingList[i][0],ratingList[i][1]
        user_privacy_weight=user_privacy_vector[user]
        item_privacy_weight=item_privacy_vector[item]
        rating_privacy_budget = user_privacy_weight*item_privacy_weight * max_budget
        if threshold > rating_privacy_budget:
            sampling_probability = (np.exp(rating_privacy_budget) -1) / (np.exp(threshold) - 1)
        else:
            sampling_probability = 1
        p = np.random.uniform(0,1)  #sample each rating w.p. sampling probability
        if p <= sampling_probability:
            sampled_index.append(i)

    sampled_ratings = [
        ratingList[i] for i in sampled_index
    ]
    return sampled_ratings


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    #special hyperparameter for sampling mechanism
    parser.add_argument('--strategy',
                         type=str,
                         default="mean",
                         help='threshold strategy for sampling mechanism')

    parser.add_argument('--mode',
                type=str,
                default="cv",
                help='cv means cross-validation mode, test means utilizing the best hyperparameter to evaluate on test ratings')
     # hyperparameter
    parser.add_argument('--embedding_dim',
						type=int,
						default=10,
						help='Embedding dimensions')
    parser.add_argument('--regularization',
                        type=float,
                        default=0.001,
                        help='L2 regularization for user and item embeddings.')
    parser.add_argument('--lr_scheme',
                        type=str,
                        default="20 40",
                        help='change epoch of lr,before 1st number lr=0.005,1st-2nd number lr=0.001, after 2nd number lr=0.0001')
    parser.add_argument('--stddev',
                        type=float,
                        default=0.1,
                        help='Standard deviation for initialization.')
    
    #experiment setting
    parser.add_argument('--data',
						type=str,
						default='Data/ml-1m',
						help='Path to the dataset')
    parser.add_argument('--max_budget',
                        type=float,
                        default=1.0,
                        help='maximum privacy budget for all the ratings')
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='Number of private training epochs in a decentralized way')
    parser.add_argument('--fraction',
                        type=str,
                        default="0.54 0.37 0.09",
                        help='fraction for 3 types of users')
    parser.add_argument('--user_privacy',
                        type=str,
                        default='0.2 0.6 1',
                        help='privacy weight list for different type of uses')
    parser.add_argument('--item_privacy',
                        type=str,
                        default="0.2 0.6 1",
                        help='privacy weight list for different type of items')
    parser.add_argument('--filename',
                        type=str,
                        default="./Results/hdp.csv",
                        help='filename to store the training and testing result')
    parser.add_argument('--logfile',
                        type=str,
                        default="./log/hdp.log",
                        help='path to store the log file')
    
    args = parser.parse_args()


    #Setting the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s -%(filename)s[line:%(lineno)d]- %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(args.logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    #Start running the main procedure
    logger.info("Start running decentralized sampling dpmf")
    logger.info(args)


    # Load the dataset
    dataset = Dataset_explicit(args.data)
    train_ratings = dataset.trainMatrix
    test_ratings = dataset.testRatings

    lr_scheme=string_to_list(args.lr_scheme)
    user_privacy_list=string_to_list(args.user_privacy)
    user_type_fraction=string_to_list(args.fraction)  
    item_privacy_list = string_to_list(args.item_privacy)

    
    try:
        if args.mode=="cv":
        #5-fold cross validation
            cv_result=[]
            X=np.arange(0,len(sampled_train_rating_matrix))
            n_splits=5
            kf=KFold(n_splits=n_splits,random_state=1,shuffle=True)
            sum_mae,sum_mse=0,0
            for train_index,test_index in kf.split(X):
                train_data=[sampled_train_rating_matrix[i] for i in train_index]
                validation_data=[sampled_train_rating_matrix[j] for j in test_index]

                user_dict,item_dict=get_user_and_item_dict(train_data)
                num_users=max(max(user_dict.values()),len(user_dict))
                num_items=max(max(item_dict.values()),len(item_dict))
                logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #val_pairs=%d' %
                    (num_users, num_items, len(train_data),len(validation_data)))

                user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_fraction,item_privacy_list)
                threshold=get_threshold(train_data,user_privacy_vector,item_privacy_vector,args.max_budget,args.strategy)
                logger.info(f"threshold in {args.strategy} strategy={threshold}")

                sampled_ratings=get_sampled_ratings(train_data,threshold,args.max_budget, user_privacy_vector,item_privacy_vector)
                logger.info(f"the size of sampled training dataset: {len(sampled_ratings)}")

                #compute the number of rated users for each item in the sampled ratings
                sampled_num_rated_users=get_num_rated_user(sampled_ratings)
                
                # Initialize the model
                model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                                args.regularization, args.stddev)

                for epoch in range(args.epochs):
                    if epoch<=lr_scheme[0]:
                        lr=0.005
                    elif epoch<=lr_scheme[1]:
                        lr=0.001
                    else:
                        lr=0.0001

                    train_mae,train_mse= model.fit(threshold,train_data,lr,sampled_num_rated_users,item_dict)
                    train_mae=round(train_mae,4)
                    train_mse=round(train_mse,4)

                    test_mae,test_mse = evaluate(model, validation_data,user_privacy_vector,item_privacy_vector)
                    test_mae=round(test_mae,4)
                    test_mse=round(test_mse,4)
                    if epoch%5==0 or epoch==args.epochs-1: 
                        logger.info('Epoch %4d:\t trainmae=%.4f\t testmae=%.4f\t trainmse=%.4f\t valmse=%.4f\t' % (epoch, train_mae,test_mae,train_mse,test_mse))
                
                sum_mae+=test_mae  #add the final mae
                sum_mse+=test_mse
            
            avg_mae=sum_mae/n_splits  
            avg_mse=sum_mse/n_splits
            cv_result.append([avg_mae,avg_mse])
            #Write cross_validation result into csv
            logger.info(f"Writing {args.mode} results into {args.filename}")
            with open(args.filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["avg_mae,avg_mse"])
                for row in cv_result:
                    writer.writerow(row)


        if args.mode=="test":
            user_dict,item_dict=get_user_and_item_dict(train_ratings)
            num_users=max(max(user_dict.values()),len(user_dict))
            num_items=max(max(item_dict.values()),len(item_dict))
            logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
                (num_users, num_items, len(train_ratings),len(test_ratings)))

            user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_fraction,item_privacy_list)
            logger.info(f"{user_privacy_vector} {item_privacy_vector}")
            threshold=get_threshold(train_ratings,user_privacy_vector,item_privacy_vector,args.max_budget,args.strategy)
            logger.info(f"threshold in {args.strategy} strategy={threshold}")

            sampled_ratings=get_sampled_ratings(train_ratings,threshold,args.max_budget,user_privacy_vector,item_privacy_vector)
            logger.info(f"the size of sampled training dataset: {len(sampled_ratings)}")

            #compute the number of rated users for each item in the sampled ratings
            sampled_num_rated_users=get_num_rated_user(sampled_ratings)
            
            # Initialize the model
            model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                            args.regularization, args.stddev)

            training_result = []
            # Train and evaluate model
            for epoch in range(args.epochs):
                if epoch<=lr_scheme[0]:
                    lr=0.005
                elif epoch<=lr_scheme[1]:
                    lr=0.001
                else:
                    lr=0.0001

                train_mae,train_mse= model.fit(threshold,sampled_ratings,lr,sampled_num_rated_users,item_dict)
                train_mae=round(train_mae,4)
                train_mse=round(train_mse,4)

                test_mae,test_mse = evaluate(model, test_ratings,user_privacy_vector,item_privacy_vector)
                test_mae=round(test_mae,4)
                test_mse=round(test_mse,4)
                logger.info('Epoch %4d:\t trainmae=%.4f\t testmae=%.4f\t trainmse=%.4f\t testmse=%.4f\t' % (epoch+1, train_mae,test_mae,train_mse,test_mse))
                training_result.append([epoch,train_mae,test_mae,train_mse,test_mse])

            logger.info(f"Writing {args.mode} results into {args.filename}")
            with open(args.filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch", "train_mae","test_mae","train_mse","test_mse"])
                for row in training_result:
                    writer.writerow(row)


    except Exception:
        logger.error('Something wrong', exc_info=True)
    logger.info('Program Finished')

if __name__ == '__main__':
    main()
