'''
@author: Wentao Hu (stevenhwt@gmail.com)
'''
import argparse
from random import triangular
from dataprocess import *
import numpy as np
import csv
import logging
from utils import *
from start_run import random_seed
np.random.seed(random_seed)



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
                prediction=1
            err_ui = rating - prediction
            absolute_loss += abs(err_ui)
            square_loss+=err_ui**2

    mae=absolute_loss/num_test_examples
    mse=square_loss/num_test_examples
    return mae,mse


def get_threshold(ratingList,user_privacy_vector,item_privacy_vector,max_budget,strategy):
    '''get the sampling threshold according to sampling stategy'''
    sum_privacy=0
    min_privacy=9999
    max_privacy=-9999
    for i in range(len(ratingList)):
        user,item=ratingList[i][0],ratingList[i][1]
        user_privacy_weight=user_privacy_vector[user]
        item_privacy_weight=item_privacy_vector[item]
        rating_privacy_budget = user_privacy_weight*item_privacy_weight * max_budget
        sum_privacy+=rating_privacy_budget
        min_privacy=min(min_privacy,rating_privacy_budget)
        max_privacy=max(max_privacy,rating_privacy_budget)
    
    if strategy=="mean":
        threshold=sum_privacy/len(ratingList)
    elif strategy=="min":
        threshold=min_privacy
    else:
        threshold=max_privacy
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
        prob_dist=[sampling_probability,1-sampling_probability]
        p = np.random.choice([1,0],1,p=prob_dist)[0] #p=1 means we choose it after sampling
        if p ==1:
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
                        help='cv means cross-validation mode, test means test mode')
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
                        default="sampling.csv",
                        help='filename to store the training and testing result')
    parser.add_argument('--logfile',
                        type=str,
                        default="sampling.log",
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
    logger.info(f"Start running decentralized sampling dpmf, random_seed={random_seed}")
    logger.info(args)


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

            #dataprocessing of sampling mechanism
            user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_ratio,item_privacy_list,item_type_ratio)
            threshold=get_threshold(train_data,user_privacy_vector,item_privacy_vector,args.max_budget,args.strategy)
            #get the sampling threshold according to sampling stategy
            logger.info(f"threshold in {args.strategy} strategy={threshold}")

            sampled_ratings=get_sampled_ratings(train_data,threshold,args.max_budget,user_privacy_vector,item_privacy_vector)
            logger.info(f"the size of sampled training dataset: {len(sampled_ratings)}")

            #compute the number of rated users for each item in the sampled ratings
            sampled_num_rated_users=get_num_rated_user(sampled_ratings)
            
            
            # Initialize the model
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

                train_mae,train_mse= model.fit(threshold,sampled_ratings,lr,sampled_num_rated_users)
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
            train_ratings=load_rating_file_as_list(f"{args.data}/u.base")
            test_ratings=load_rating_file_as_list(f"{args.data}/u.test")
            logger.info(f"dataset: {args.data}/u.base {args.data}/u.test")
            user_dict,item_dict=get_user_and_item_dict(train_ratings)

            num_users=max(max(user_dict.values()),len(user_dict))
            num_items=max(max(item_dict.values()),len(item_dict))
            logger.info('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
                (num_users, num_items, len(train_ratings),len(test_ratings)))

            user_privacy_vector,item_privacy_vector=get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_ratio,item_privacy_list,item_type_ratio)
            threshold=get_threshold(train_ratings,user_privacy_vector,item_privacy_vector,args.max_budget,args.strategy)
            #get the sampling threshold according to sampling stategy
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
            for epoch in range(total_epochs):
                if epoch<=0.25*total_epochs:
                    lr=init_lr
                elif epoch<=0.75*total_epochs:
                    lr=init_lr/5
                else:
                    lr=init_lr/25

                train_mae,train_mse= model.fit(threshold,sampled_ratings,lr,sampled_num_rated_users)
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