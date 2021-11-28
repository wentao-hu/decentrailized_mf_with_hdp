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

class MFModel(object):
    """A matrix factorization model trained using SGD."""
    def __init__(self, num_user, num_item, embedding_dim, reg, stddev):
        self.user_embedding = np.random.normal(0, stddev,(num_user, embedding_dim))
        self.item_embedding = np.random.normal(0, stddev,(num_item, embedding_dim))
        self.reg = reg
        self.embedding_dim = embedding_dim

    def _predict_one(self, user, item):
        """Predicts the score of a user for an item."""
        prediction=np.dot(self.user_embedding[user], self.item_embedding[item])
        return prediction

    def fit(self, train_rating_matrix,learning_rate):
        np.random.shuffle(train_rating_matrix)

        num_examples = len(train_rating_matrix)
        reg = self.reg
        embedding_dim = self.embedding_dim
        lr = learning_rate
        absolute_loss = 0.0
        square_loss=0

        #nonprivate update
        for i in range(num_examples):
            (user, item, rating) = train_rating_matrix[i]
            user_emb = self.user_embedding[user]
            item_emb = self.item_embedding[item]

            prediction = self._predict_one(user, item)
            err_ui = rating - prediction
            
            absolute_loss += abs(err_ui)   
            square_loss += err_ui**2

            self.user_embedding[user,:] += lr * 2 * (err_ui * item_emb - reg * user_emb)
            self.item_embedding[item,:]+=lr*2*(err_ui*user_emb-reg*item_emb)

        mae=absolute_loss/num_examples
        mse=square_loss/num_examples
        return mae,mse


def evaluate(model, test_ratings,user_dict,item_dict):
    '''Evaluate loss on the test dataset'''
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
            if prediction>=5:
                prediction=5
            if prediction<=1:
                prediction=1
            err_ui = rating - prediction
            absolute_loss += abs(err_ui)
            square_loss+=err_ui**2
    
    mae=absolute_loss/num_test_examples
    mse=square_loss/num_test_examples
    return mae,mse



def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    #running mode
    parser.add_argument('--mode',
						type=str,
						default="cv",
						help='cv means cross-validation mode, test means using the best hyperparameter to evaluate on test ratings')

    #hyperparameter setting
    parser.add_argument('--lr_scheme',
                        type=str,
                        default="20 40",
                        help='change epoch of lr,before 1st number lr=0.005,1st-2nd number lr=0.001, after 2nd number lr=0.0001')  
    parser.add_argument('--embedding_dim',
						type=int,
						default=10,
						help='Embedding dimensions')
    parser.add_argument('--regularization',
                        type=float,
                        default=0.001,
                        help='L2 regularization for user and item embeddings.')             
    parser.add_argument('--stddev',
                        type=float,
                        default=0.01,
                        help='Standard deviation for initialization.')

    #experiment setting
    parser.add_argument('--data',
						type=str,
						default='Data/ml-100k',
						help='Path to the dataset') 
    parser.add_argument('--epochs',
                        type=int,
                        default=60,
                        help='Number of private training epochs in a decentralized way')
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

    lr_scheme=string_to_list(args.lr_scheme)
    try:
        if args.mode=="cv":
            #5-fold cross validation
            cv_result=[] 
            sum_mae,sum_mse=0,0
            for i in range(1,6):
                logger.info(f"dataset: {args.data}/u{i}.base  {args.data}/u{i}.test")
                #load data
                train_data=load_rating_file_as_list(f"{args.data}/u{i}.base")
                validation_data=load_rating_file_as_list(f"{args.data}/u{i}.test")

                user_dict,item_dict=get_user_and_item_dict(train_data) 
                num_users=max(max(user_dict.values()),len(user_dict))
                num_items=max(max(item_dict.values()),len(item_dict))
                logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #validation_pairs=%d' %
                    (num_users, num_items, len(train_data),len(validation_data)))

                # Initialize the model, need to plus 1 because the index start from 0
                model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                                args.regularization, args.stddev)

                #train the model
                for epoch in range(args.epochs):
                    if epoch<=lr_scheme[0]:
                        lr=0.005
                    elif epoch<=lr_scheme[1]:
                        lr=0.001
                    else:
                        lr=0.0001

                    train_mae,train_mse= model.fit(train_data,lr)
                    train_mae=round(train_mae,4)
                    train_mse=round(train_mse,4)

                    test_mae,test_mse = evaluate(model, validation_data,user_dict,item_dict)
                    test_mae=round(test_mae,4)
                    test_mse=round(test_mse,4)
                    logger.info('Epoch %4d:\t trainmae=%.4f\t valmae=%.4f\t trainmse=%.4f\t valmse=%.4f\t' % (epoch, train_mae,test_mae,train_mse,test_mse))
                
                cv_result.append([i,test_mae,test_mse])
                sum_mae+=test_mae  #add the final mae
                sum_mse+=test_mse

            avg_mae=sum_mae/5  
            avg_mse=sum_mse/5
            cv_result.append(["avg",avg_mae,avg_mse])
            #Write cross_validation result into csv
            logger.info(f"Writing {args.mode} results into {args.filename}")
            with open(args.filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["round","valmae","valmse"])
                for row in cv_result:
                    writer.writerow(row)



        if args.mode=="test":
            #load data
            train_ratings=load_rating_file_as_list(f"{args.data}/ua.base")
            test_ratings=load_rating_file_as_list(f"{args.data}/ua.test")   
            user_dict,item_dict=get_user_and_item_dict(train_ratings)

            num_users=max(max(user_dict.values()),len(user_dict))
            num_items=max(max(item_dict.values()),len(item_dict))
            logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
                (num_users, num_items, len(train_ratings),len(test_ratings)))

            # Initialize the model
            model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                            args.regularization, args.stddev)
            # Train and evaluate model
            training_result = []
            for epoch in range(args.epochs):
                if epoch<=lr_scheme[0]:
                    lr=0.005
                elif epoch<=lr_scheme[1]:
                    lr=0.001
                else:
                    lr=0.0001

                train_mae,train_mse= model.fit(train_ratings,lr)
                train_mae=round(train_mae,4)
                train_mse=round(train_mse,4)

                test_mae,test_mse = evaluate(model,test_ratings,user_dict,item_dict)
                test_mae=round(test_mae,4)
                test_mse=round(test_mse,4)
                logger.info('Epoch %4d:\t trainmae=%.4f\t testmae=%.4f\t trainmse=%.4f\t testmse=%.4f\t' % (epoch+1, train_mae,test_mae,train_mse,test_mse))
                training_result.append([epoch+1,train_mae,test_mae,train_mse,test_mse])


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
