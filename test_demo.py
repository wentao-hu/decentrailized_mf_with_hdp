import argparse
from random import triangular
from dataprocess import *
import numpy as np
import random
import csv
import logging 
from sklearn.model_selection import KFold
from utils_private import *
from mf_sampling_decentralized import *
from mf_hdp_decentralized import *
import pandas as pd

data="Data/ml-1m"
train_data=load_rating_file_as_list(f"{data}/u.base")
print(train_data[:10])

