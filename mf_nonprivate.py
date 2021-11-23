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
        """Predicts the score of a user for an item."""
        raw_prediction=np.dot(self.user_embedding[user], self.item_embedding[item])
        if raw_prediction>=5:
            prediction=5
        elif raw_prediction<=1:
            prediction=1
        else:
            prediction=raw_prediction
        return prediction

    def fit(self, train_rating_matrix,learning_rate):
        """Nonprivately Train the model for one epoch.
		Args:
			train_rating_matrix: a nested list, each secondary list representing a positive
				user-item pair and their rating.
			learning_rate: the learning rate to use.

		Returns:
			The absolute loss averaged across examples.
		"""
        np.random.shuffle(train_rating_matrix)

        # Iterate over all examples and perform one SGD step.
        num_examples = len(train_rating_matrix)
        reg = self.reg
        embedding_dim = self.embedding_dim
        lr = learning_rate
        absolute_loss = 0.0
        square_loss=0

        #On user device, user embedding is updated normally by users
        for i in range(num_examples):
            (user, item, rating) = train_rating_matrix[i]
            user_emb = self.user_embedding[user]
            item_emb = self.item_embedding[item]

            prediction = self._predict_one(user, item)
            err_ui = rating - prediction
            
            absolute_loss += abs(err_ui)   
            square_loss+=(err_ui)**2

            self.user_embedding[user,:] += lr * 2 * (err_ui * item_emb - reg * user_emb)
            self.item_embedding[item,:]+=lr*2*(err_ui*user_emb-reg*item_emb)

        mae=absolute_loss/num_examples
        mse=square_loss/num_examples
        return mae,mse


def evaluate(model, test_ratings,user_dict,item_dict):
    '''Evaluate loss on the test dataset,sampling method does not need to scale back '''
    absolute_loss = 0
    square_loss=0
    num_test_examples=0
    train_users=user_dict.values()
    train_items=item_dict.values()
    for i in range(len(test_ratings)):
        (user, item, rating) = test_ratings[i]
        #only consider users and items appear in training dataset
        if user in train_users and item in train_items:
            num_test_examples+=1
            prediction = model._predict_one(user, item)
            err_ui = rating - prediction
            absolute_loss += abs(err_ui)
            square_loss+=err_ui**2
    
    mae=absolute_loss/num_examples
    mse=square_loss/num_examples
    return mae,mse

def string_to_list(str):
    '''transfer a numerical string to list, used in parse some arguments below'''
    tmp=str.split(" ")
    lst=[float(x) for x in tmp]
    return lst

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    #experiment setting
    parser.add_argument('--data',
						type=str,
						default='Data/ml-1m',
						help='Path to the dataset')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of private training epochs in a decentralized way')
    
    #hyperparameter setting
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
    logger.info("Start running nonprivate mf")
    logger.info(args)


    # Load the dataset
    dataset = Dataset_explicit(args.data)
    train_rating_matrix = dataset.trainMatrix
    test_ratings = dataset.testRatings
    user_dict=dataset.user_dict
    item_dict=dataset.item_dict
    

    num_users=max(max(user_dict.values()),len(user_dict))
    num_items=max(max(item_dict.values()),len(item_dict))
    logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
          (num_users, num_items, len(train_rating_matrix),len(test_ratings)))
    
    # Initialize the model
    model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                    args.regularization, args.stddev)

    training_result = []
    # Train and evaluate model
    try:
        for epoch in range(args.epochs):
            train_mae,train_mse= model.fit(train_rating_matrix,args.learning_rate)
            train_mae=round(train_mae,4)
            train_mse=round(train_mse,4)

            test_mae,test_mse = evaluate(model, test_ratings,user_dict,item_dict)
            test_mae=round(test_mae,4)
            test_mse=round(test_mse,4)
            logger.info('Epoch %4d:\t trainmae=%.4f\t testmae=%.4f\t trainmse=%.4f\t testmse=%.4f\t' % (epoch+1, train_mae,test_mae,train_mse,test_mse))
            training_result.append([epoch+1,train_mae,test_mae,train_mse,test_mse])

        #Write the training result into csv
        logger.info(f"Writing results into {args.filename}")
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
