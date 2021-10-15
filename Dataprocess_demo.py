from Dataset_explicit import Dataset_explicit
import numpy as np

import os
import csv

training_result=[[1,2,3]]
with open("./Results/test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["privacy_mode", "epoch", "mse"])
        for row in training_result:
            writer.writerow(row)
