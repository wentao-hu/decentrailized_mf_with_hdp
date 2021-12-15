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
from utils import *
np.random.seed(2)


#For evaluating on test ratings in HDP mechanism
def evaluate(model, test_ratings,user_privacy_vector,item_privacy_vector):
    '''Evaluate loss on the test dataset , in HDP we should scale back using user_privacy_vector and item_privacy_vector'''
    absolute_loss = 0
    square_loss=0
    num_test_examples=0
    train_users=user_privacy_vector.keys()
    train_items=item_privacy_vector.keys()
    for i in range(len(test_ratings)):
        (user, item, rating) = test_ratings[i]
        #only consider users and items appear in training dataset,eliminate only 2-4 records
        if user in train_users and item in train_items:
            num_test_examples+=1
            raw_prediction=np.dot(model.user_embedding[user]/user_privacy_vector[user], model.item_embedding[item]/item_privacy_vector[item])
            if raw_prediction>=5:
                prediction=5
            elif raw_prediction<=1:
                prediction=1
            else:
                prediction=raw_prediction
        err_ui= rating - prediction
        absolute_loss += abs(err_ui)
        square_loss += err_ui**2

    mae=absolute_loss/num_test_examples
    mse=square_loss/num_test_examples
    return mae,mse


#For stretching rating in HDP mechanism
def stretch_rating(ratingList,user_privacy_vector,item_privacy_vector):
    # stretch the train rating matrix
    for i in range(len(ratingList)):
        user,item=ratingList[i][0],ratingList[i][1]
        user_privacy_weight=user_privacy_vector[user]
        item_privacy_weight=item_privacy_vector[item]
        rating_privacy_weight = user_privacy_weight*item_privacy_weight
        ratingList[i][2] *= rating_privacy_weight
    return ratingList


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    #running mode 
    parser.add_argument('--mode',
						type=str,
						default='test',
						help='cv means cross validation mode, test means test mode')
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
                        default=100,
                        help='Number of private training epochs in a decentralized way')
    parser.add_argument('--user_ratio',
                        type=str,
                        default="0.54 0.37 0.09",
                        help='ratio for 3 types of users')
    parser.add_argument('--item_ratio',
                        type=str,
                        default="0.3333 0.3333 0.3334",
                        help='ratio for 3 types of items')
    parser.add_argument('--user_privacy',
                        type=str,
                        default='0.1 0.5 1',
                        help='privacy weight list for different type of uses')
    parser.add_argument('--item_privacy',
                        type=str,
                        default="0.1 0.5 1",
                        help='privacy weight list for different type of items')
    parser.add_argument('--filename',
                        type=str,
                        default="./Results/hdp.csv",
                        help='filename to store the training and testing result')
    parser.add_argument('--logfile',
                        type=str,
                        default="./log/hdp.log",
                        help='path to store the log file')

    # hyperparameter
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='initial learning rate') 
    parser.add_argument('--embedding_dim',
						type=int,
						default=10,
						help='Embedding dimensions')
    parser.add_argument('--regularization',
                        type=float,
                        default=0.01,
                        help='L2 regularization for user and item embeddings.')             
    parser.add_argument('--stddev',
                        type=float,
                        default=0.1,
                        help='Standard deviation for initialization.')    
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
    logger.info(args)

    # Load the dataset  
    user_privacy_list=string_to_list(args.user_privacy)   
    item_privacy_list = string_to_list(args.item_privacy)
    user_type_ratio=string_to_list(args.user_ratio) 
    item_type_ratio=string_to_list(args.item_ratio)
    init_lr=args.lr
    total_epochs=args.epochs

    try:
        if args.mode=="cv":
             # randomly choose 1 validation set to do validation
            i=np.random.choice([1,2,3,4,5],1)[0]
          
            logger.info(f"dataset: {args.data}/u{i}.base  {args.data}/u{i}.test")
            train_data=load_rating_file_as_list(f"{args.data}/u{i}.base")
            validation_data=load_rating_file_as_list(f"{args.data}/u{i}.test")

            user_dict,item_dict=get_user_and_item_dict(train_data) 
            num_users=max(max(user_dict.values()),len(user_dict))
            num_items=max(max(item_dict.values()),len(item_dict))
            logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #validation_pairs=%d' %
                (num_users, num_items, len(train_data),len(validation_data)))

            #get privacy vector and stretched rating
            user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_ratio,item_privacy_list,item_type_ratio)
            stretch_ratings=stretch_rating(train_data,user_privacy_vector,item_privacy_vector)
            num_rated_users=get_num_rated_user(train_data)
            # Initialize the model, need to plus 1 because the index start from 0
            model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                            args.regularization, args.stddev)

            cv_result=[]
            for epoch in range(total_epochs):
                if epoch<=0.25*total_epochs:
                    lr=init_lr
                elif epoch<=0.75*total_epochs:
                    lr=init_lr/5
                else:
                    lr=init_lr/25

                train_mae,train_mse= model.fit(args.max_budget,stretch_ratings,lr,num_rated_users)
                train_mae=round(train_mae,4)
                train_mse=round(train_mse,4)

                test_mae,test_mse = evaluate(model, validation_data,user_privacy_vector,item_privacy_vector)
                test_mae=round(test_mae,4)
                test_mse=round(test_mse,4)
                logger.info('Epoch %4d:\t trainmae=%.4f\t valmae=%.4f\t trainmse=%.4f\t valmse=%.4f\t' % (epoch+1, train_mae,test_mae,train_mse,test_mse))
                cv_result.append([epoch+1,train_mae,test_mae,train_mse,test_mse])

            
            #Write cross_validation result into csv
            logger.info(f"Writing {args.mode} results into {args.filename}")
            with open(args.filename, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["epoch", "train_mae","val_mae","train_mse","val_mse"])
                for row in cv_result:
                    writer.writerow(row)



        if args.mode=="test":
            train_ratings=load_rating_file_as_list(f"{args.data}/ub.base")
            test_ratings=load_rating_file_as_list(f"{args.data}/ub.test")
            logger.info(f"dataset: {args.data}/u.base {args.data}/u.test")
            user_dict,item_dict=get_user_and_item_dict(train_ratings)

            num_users=max(max(user_dict.values()),len(user_dict))
            num_items=max(max(item_dict.values()),len(item_dict))
            logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
                (num_users, num_items, len(train_ratings),len(test_ratings)))

            #get privacy vector and stretched rating
            user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_ratio,item_privacy_list,item_type_ratio)
            stretch_ratings=stretch_rating(train_ratings,user_privacy_vector,item_privacy_vector)
            
            num_rated_users=get_num_rated_user(train_ratings)
            # Initialize the model, need to plus 1 because the index start from 0
            model = MFModel(num_users+1, num_items+1, args.embedding_dim,
                            args.regularization, args.stddev)

            training_result = []
            for epoch in range(total_epochs):
                if epoch<=0.25*total_epochs:
                    lr=init_lr
                elif epoch<=0.75*total_epochs:
                    lr=init_lr/5
                else:
                    lr=init_lr/25

                train_mae,train_mse= model.fit(args.max_budget,stretch_ratings,lr,num_rated_users)
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