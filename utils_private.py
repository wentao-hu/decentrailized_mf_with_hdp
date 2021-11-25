import numpy as np
import random

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
        prediction=np.dot(self.user_embedding[user], self.item_embedding[item])
        return prediction

    def fit(self, privacy_budget, train_rating_matrix,learning_rate,num_rated_users,item_dict):
        """Trains the model for one epoch.

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
        Delta = 4
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
            square_loss+=err_ui**2

            self.user_embedding[user,:] += lr * 2 * (err_ui * item_emb - reg * user_emb)

        
        #On server side, item embedding is updated aftering gathering private gradient from all users
        h_dict={}    #get h_j when updating item embedding v_j
        for j in range(len(item_dict)):
            item=item_dict[j]
            h_dict[item]=np.random.exponential(1,embedding_dim) 

        for i in range(num_examples):
            (user, item, rating) = train_rating_matrix[i]
            user_emb = self.user_embedding[user]
            item_emb = self.item_embedding[item]

            prediction = self._predict_one(user, item)
            err_ui = rating - prediction

            std=np.sqrt(1/num_rated_users[item])
            c=np.random.normal(0,std,embedding_dim)
            h=h_dict[item]
            noise_vector=2 * Delta*np.sqrt(2*embedding_dim*h) *c/privacy_budget

            self.item_embedding[item,:]+=lr*(2*(err_ui*user_emb-reg*item_emb)-noise_vector)

        mae=absolute_loss/num_examples
        mse=square_loss/num_examples
        return mae,mse


#get privacy vector according to fraction of different users and items , and their privacy weight
def get_privacy_vector(user_dict,item_dict,user_privacy_list,user_type_fraction,item_privacy_list):
    user_privacy_vector={}
    item_privacy_vector={}
    for i in range(len(user_dict)):
        tmp=np.random.choice(user_privacy_list,1,p=user_type_fraction)  #return a list at length 1
        user_privacy_weight=tmp[0]
        user=user_dict[i]
        user_privacy_vector[user]=user_privacy_weight
    
    for j in range(len(item_dict)):
        item_privacy_weight=random.choice(item_privacy_list)
        item=item_dict[j]
        item_privacy_vector[item]=item_privacy_weight
    return user_privacy_vector,item_privacy_vector