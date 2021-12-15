import numpy as np
import pandas as pd
np.random.seed(2)

# p1=np.random.choice([1,2,3,4,5],1)
# print(p1)

for j in range(2):
    print(j)
    for i in range(20):
        print(np.random.choice([0,1],1,p=[0.3,0.7]))