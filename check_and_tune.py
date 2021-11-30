'''
author: wentao hu (stevenhwt@gmail.com)
input: some folder that stores the cross-validaiton results
output: the best hyperparameter combination in this folder
'''

import os
import re


data="ml-1m"
method=""
path=f"Data/log-ml-1m"

for filename in os.listdir