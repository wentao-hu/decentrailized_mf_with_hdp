from Dataset_explicit import Dataset_explicit
import numpy as np

import os
import csv
import argparse

training_result=[[1,2,3]]
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='Results',
                      help='Path to the dataset')
args = parser.parse_args()
t=1
with open(f"./{args.file}/test-threshold={t}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["privacy_mode", "epoch", "mse"])
        for row in training_result:
            writer.writerow(row)
