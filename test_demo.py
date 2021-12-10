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

v1=np.random.rand(10)
v2=np.random.rand(10)
sum=v1.dot(v2)
print(sum)