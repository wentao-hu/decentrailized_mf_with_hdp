import numpy as np
import pandas as pd

for i in [0,1,2,3,10]:
    np.random.seed(i)
    print(np.random.choice([1,2,3,4,5],1))
