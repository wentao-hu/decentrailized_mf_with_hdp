import os
import time
import numpy as np

#define bash command and write into .sh file
str1="""
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/sampling/%J.err 
#BSUB -o ./log/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
"""

#self-defined command for hyperparameter tuning 
lr_scheme="20 40"
for embedding_dim in [2,3,4,5,10]:
    filename=f"./Results/sampling/sampling_dim={embedding_dim}.csv"
    logfile=f"./log/sampling/sampling_dim={embedding_dim}.log"

    str2=f""" python mf_sampling_decentralized.py --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """ 

    with open('run_sampling_decentralized.sh','w') as f:   
        f.write(str1+str2)

    #run .sh file
    cmd = 'bsub < run_sampling_decentralized.sh'
    os.system(cmd)
    time.sleep(10)