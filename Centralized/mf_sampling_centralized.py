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
Modified on Oct 12, 2021
1. Delete the bias in the model
2. Change logistic loss to square loss for explicit feedback 
3. Change the evaluation metrics from HR and NDCG to MSE
4. Add Laplacian noise to the gradient in a centralized way
5. Reproduce the experiments in "Probabilistic matrix factorization with personalized differential
privacy,KBS 2019,Zhang et al."
@author: Wentao Hu (stevenhwt@gmail.com)
'''

import argparse
from random import triangular
from Dataset_explicit import Dataset_explicit
import numpy as np
import random
import csv


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

    def fit(self, private_mode, privacy_budget, train_rating_matrix,learning_rate):
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
        l1_sensitivity = 2 * np.sqrt(embedding_dim) * Delta
        sum_of_loss = 0.0
        for i in range(num_examples):
            (user, item, rating) = train_rating_matrix[i]
            prediction = self._predict_one(user, item)

            err_ui = rating - prediction
            if private_mode == 0:
                for k in range(embedding_dim):
                    #udpate user_embedding
                    self.user_embedding[user,k] += lr * 2 * (err_ui * self.item_embedding[item][k] -
                                        reg * self.user_embedding[user, k])
                    #update item_embedding
                    self.item_embedding[item,k] += lr * 2 * (err_ui * self.user_embedding[user][k] -
                                        reg * self.item_embedding[item, k])
            else:
                #In private update k2, we only need to update item embedding once to make it contains less private information
                for k in range(embedding_dim):
                    self.item_embedding[item, k] += lr * (2 * (err_ui * self.user_embedding[user][k] - reg *self.item_embedding[item, k]) 
                    - np.random.laplace(0, l1_sensitivity / privacy_budget, 1))
                    #the latter is negative grad minus laplace noise
            sum_of_loss += err_ui**2

        # Return the mean square loss during training process.
        return sum_of_loss / num_examples


def evaluate(model, test_ratings):
    '''Evaluate MSE on the test dataset '''
    square_loss = 0
    for i in range(len(test_ratings)):
        (user, item, rating) = test_ratings[i]
        prediction = model._predict_one(user, item)
        err = rating - prediction
        square_loss += err**2
    num_test_examples = len(test_ratings)
    return square_loss / num_test_examples


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        '--data',
                        type=str,
                        default='Data/ml-1m',
                        help='Path to the dataset')
    parser.add_argument('--max_budget',
                        type=float,
                        default=1.0,
                        help='maximum privacy budget for all the ratings')

    parser.add_argument('--nonprivate_epochs',
                        type=int,
                        default=54,
                        help='Number of non-private training epochs')

    parser.add_argument('--private_epochs',
                        type=int,
                        default=54,
                        help='Number of non-private training epochs')

    parser.add_argument(
                        '--embedding_dim',
                        type=int,
                        default=8,
                        help='Embedding dimensions, the first dimension will be '
                        'used for the bias.')
    parser.add_argument('--regularization',
                        type=float,
                        default=0.0,
                        help='L2 regularization for user and item embeddings.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='SGD step size.')
    parser.add_argument('--stddev',
                        type=float,
                        default=0.1,
                        help='Standard deviation for initialization.')
    parser.add_argument('--threshold',
                        type=float,
                        default=1,
                        help='The threshold in the sampling mechanism')
    args = parser.parse_args()

    # Load the dataset
    dataset = Dataset_explicit(args.data)
    train_rating_matrix = dataset.trainMatrix
    test_ratings = dataset.testRatings
    print('Dataset: #user=%d, #item=%d, #train_pairs=%d, #test_pairs=%d' %
          (dataset.num_users, dataset.num_items, len(train_rating_matrix),
           len(test_ratings)))

    # Data processing
    #Determine the heterogeneous privacy weight for each trainning rating
    # Sampling the rating with heterogeneous probability
    user_level_list = [0.3, 0.6, 0.9]
    item_level_list = [0.2, 0.4, 0.8]
    threshold = args.threshold
    sampled_index = []
    num_training_examples = len(train_rating_matrix)
    max_budget = args.max_budget
    for i in range(num_training_examples):
        user_level_weight = random.choice(user_level_list)
        item_level_weight = random.choice(item_level_list)
        rating_privacy_budget = user_level_weight * item_level_weight * max_budget
        if threshold > rating_privacy_budget:
            sampling_probability = (np.exp(rating_privacy_budget) -
                                    1) / (np.exp(threshold) - 1)
        else:
            sampling_probability = 1
        p = np.random.uniform(0,1)  #sample each rating w.p. sampling probability
        if p <= sampling_probability:
            sampled_index.append(i)

    sampled_train_rating_matrix = [
        train_rating_matrix[i] for i in sampled_index
    ]

    # Initialize the model
    model = MFModel(dataset.num_users, dataset.num_items, args.embedding_dim,
                    args.regularization, args.stddev)

    training_result = []
    # Train and evaluate model
    for epoch in range(args.nonprivate_epochs):
        # Non_private Training
        private_mode = 0
        train_mse= model.fit(private_mode,args.threshold,sampled_train_rating_matrix,args.learning_rate)
        train_mse=round(train_mse,4)

        # Evaluation
        test_mse = round(evaluate(model, test_ratings),4)
        print('Nonprivate Epoch %4d:\t trainloss=%.4f\t testloss=%.4f\t' % (epoch+1, train_mse,test_mse))
        training_result.append(["Nonprivate",epoch+1, train_mse,test_mse])

    for epoch in range(args.private_epochs):
        # Private Training
        private_mode = 1
        train_mse= model.fit(private_mode,args.threshold,sampled_train_rating_matrix,args.learning_rate)
        train_mse=round(train_mse,4)

        # Evaluation
        test_mse = round(evaluate(model, test_ratings),4)
        print('Private Epoch %4d:\t trainloss=%.4f\t testloss=%.4f\t' % (epoch+1, train_mse,test_mse))
        training_result.append(["Private", epoch+1, train_mse,test_mse])

    #Write the training result into csv
    with open(f"./Results/sampling_centralized_maxbudget={args.max_budget}_threshold={args.threshold}_nonprivnum={args.nonprivate_epochs}_privnum={args.private_epochs}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["privacy_mode", "epoch", "train_mse","test_mse"])
        for row in training_result:
            writer.writerow(row)

if __name__ == '__main__':
    main()
