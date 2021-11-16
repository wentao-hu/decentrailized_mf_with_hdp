# coding=utf-8#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''
@author: Wentao Hu (stevenhwt@gmail.com)
'''

import argparse
from random import triangular
from dataprocess import Dataset_explicit
import numpy as np
import random
import csv
import logging 


class MFModel(object):
    """A matrix factorization model trained using SGD."""
    def __init__(self, num_user, num_item, embedding_dim, reg, stddev):
        """Initializes MFModel.

		Args:
			num_user: the total number of users.
			num_item: the total number of items.
			embedding_dim: the embedding dimension.
			reg: the regularization coefficient.
			stddev: embeddings are initialized from a random distribution with this
				standard deviation.
		"""
        self.user_embedding = np.random.normal(0, stddev,(num_user, embedding_dim))
        self.item_embedding = np.random.normal(0, stddev,(num_item, embedding_dim))
        self.reg = reg
        self.embedding_dim = embedding_dim

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item"""
        raw_prediction=np.dot(self.user_embedding[user], self.item_embedding[item])
        if raw_prediction>=5:
            prediction=5
        elif raw_prediction<=0:
            prediction=0
        else:
            prediction=raw_prediction
        return prediction

    def fit(self, privacy_budget, train_rating_matrix,learning_rate,num_rated_users):
        """Trains the model for one epoch.

		Args:
			train_rating_matrix: a nested list, each secondary list representing a positive
				user-item pair and their rating.
			learning_rate: the learning rate to use.

		Returns:
			The square loss averaged across examples.
		"""
        np.random.shuffle(train_rating_matrix)

        # Iterate over all examples and perform one SGD step.
        num_examples = len(train_rating_matrix)
        reg = self.reg
        embedding_dim = self.embedding_dim
        lr = learning_rate
        Delta = 4
        sum_of_loss = 0.0

        h=np.random.exponential(1) #update once in one iteration
        for i in range(num_examples):
            (user, item, stretch_rating) = train_rating_matrix[i]
            prediction = self._predict_one(user, item)
            err_ui = stretch_rating - prediction


            std=np.sqrt(1/num_rated_users[item])
            c=np.random.normal(0,std)
            #In decentralized setting, we must privately update item embedding
            for k in range(embedding_dim):
                self.item_embedding[item, k] += lr * (2 * (err_ui * self.user_embedding[user][k] - reg *self.item_embedding[item, k]) -
                2 * Delta*np.sqrt(2*embedding_dim*h) *c/privacy_budget)
            sum_of_loss += err_ui**2

        # Return MSE on the training dataset
        return sum_of_loss / num_examples



def evaluate(model, test_ratings,user_privacy_vector,item_privacy_vector):
    '''Evaluate MSE on the test dataset , in HDP we should scale back using user_privacy_vector and item_privacy_vector'''
    square_loss = 0
    num_test_examples=0
    for i in range(len(test_ratings)):
        (user, item, rating) = test_ratings[i]
        if user in user_privacy_vector.keys() and item in item_privacy_vector.keys():
            num_test_examples+=1
            raw_prediction=np.dot(model.user_embedding[user]/user_privacy_vector[user], model.item_embedding[item]/item_privacy_vector[item])
            if raw_prediction>=5:
                prediction=5
            elif raw_prediction<=1:
                prediction=1
            else:
                prediction=raw_prediction
        err = rating - prediction
        square_loss += err**2
    return square_loss / num_test_examples


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
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
                        default=128,
                        help='Number of private training epochs in a decentralized way')
    parser.add_argument('--embedding_dim',
						type=int,
						default=10,
						help='Embedding dimensions')
    parser.add_argument('--regularization',
                        type=float,
                        default=0.001,
                        help='L2 regularization for user and item embeddings.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='SGD step size.')
    parser.add_argument('--stddev',
                        type=float,
                        default=0.1,
                        help='Standard deviation for initialization.')
    parser.add_argument('--fraction',
                        type=list,
                        default=[0.54,0.37,0.09],
                        help='fraction for 3 types of users')
    parser.add_argument('--bound',
                        type=list,
                        default=[0.1,0.2,1],
                        help='privacy budget lower bounds for 3 user types in ascending order')
    parser.add_argument('--item_privacy',
                        type=list,
                        default=[0.2, 0.6, 1],
                        help='privacy budget list for different type of items')
    parser.add_argument('--filename',
                        type=str,
                        default="./Results/hdp.csv",
                        help='filename to store the training and testing result')
    parser.add_argument('--logfile',
                        type=str,
                        default="./log/hdp.log",
                        help='filename to store the training and testing result')
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
    logger.info("Start running decentralized hdpmf")
    param_setting=f"""data={args.data} max_budget={args.max_budget} epoch={args.epochs} dim={args.embedding_dim}\
    reg={args.regularization} lr={args.learning_rate} stddev={args.stddev} frac={args.fraction} bound={args.bound}\
    item_privacy={args.item_privacy} results={args.filename} log={args.logfile}"""
    logger.info(param_setting)

    # Load the dataset
    dataset = Dataset_explicit(args.data)
    train_rating_matrix= dataset.trainMatrix
    test_ratings = dataset.testRatings
    train_num_rated_users=dataset.train_num_rated_users
    user_dict=dataset.user_dict
    item_dict=dataset.item_dict
  
    num_users=max(max(user_dict.values()),len(user_dict))
    num_items=max(max(item_dict.values()),len(item_dict))
    logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
          (num_users, num_items, len(train_rating_matrix),len(test_ratings)))


    #Determine the heterogeneous privacy weight for each trainning rating
    user_type_list= ["conservative", "moderate", "liberal"]
    user_type_fraction=args.fraction  #fraction for 3 types of users
    user_type_bound=args.bound  #privacy budget lower bounds for 3 user types in ascending order
    item_privacy_list = args.item_privacy

    user_privacy_vector={}
    item_privacy_vector={}
    for i in range(len(user_dict)):
        user_type=np.random.choice(user_type_list,1,p=user_type_fraction)
        if user_type=="conservative":
            user_privacy_weight=random.uniform(user_type_bound[0],user_type_bound[1])
        elif user_type=="moderate":
            user_privacy_weight=random.uniform(user_type_bound[1],user_type_bound[2])
        else:
            user_privacy_weight=1
        user=user_dict[i]
        user_privacy_vector[user]=user_privacy_weight
    
    for j in range(len(item_dict)):
        item_privacy_weight=random.choice(item_privacy_list)
        item=item_dict[j]
        item_privacy_vector[item]=item_privacy_weight

    # stretch the rating as the pre-processing step
    for i in range(len(train_rating_matrix)):
        user,item=train_rating_matrix[i][0],train_rating_matrix[i][1]
        user_privacy_weight=user_privacy_vector[user]
        item_privacy_weight=item_privacy_vector[item]
        rating_privacy_weight = user_privacy_weight*item_privacy_weight
        train_rating_matrix[i][2] *= rating_privacy_weight

    # Initialize the model, need to plus 1 because the index start from 0
    model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                    args.regularization, args.stddev)

    #Train and evaluate the model
    training_result = []

    try:
        for epoch in range(args.epochs):
            train_mse = model.fit(args.max_budget,train_rating_matrix,args.learning_rate,train_num_rated_users)
            train_mse=round(train_mse,4)

            # Evaluation
            test_mse = round(evaluate(model, test_ratings,user_privacy_vector,item_privacy_vector),4)
            logger.info('Epoch %4d:\t trainloss=%.4f\t testloss=%.4f\t' % (epoch+1, train_mse,test_mse))
            training_result.append([epoch+1, train_mse,test_mse])

        #Write the training result into csv
        logger.info(f"Writing results into {args.filename}")
        with open(args.filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_mse","test_mse"])
            for row in training_result:
                writer.writerow(row)
    except Exception:
        logger.error('Something wrong', exc_info=True)
    logger.info('Program Finished')
    

if __name__ == '__main__':
    main()