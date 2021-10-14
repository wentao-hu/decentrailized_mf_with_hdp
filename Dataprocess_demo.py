from Dataset_explicit import Dataset_explicit
from evaluate import evaluate_model
import numpy as np

dataset = Dataset_explicit('/home/sufe/Desktop/neural_collaborative_filtering/Data/ml-1m')
train_rating_matrix = dataset.trainMatrix
test_ratings= dataset.testRatings

print(type(test_ratings))
print(test_ratings[0])

print(type(train_rating_matrix))
print(train_rating_matrix[0])

