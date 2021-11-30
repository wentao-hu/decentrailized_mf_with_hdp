'''
author: wentao hu (stevenhwt@gmail.com)
input: some folder that stores the cross-validaiton results
output: the best hyperparameter combination in this folder
'''

import os
import re


data="ml-1m"
method="hdp"
path=f"log-{data}/{method}"

for filename in os.listdir(path):
    print(filename)
