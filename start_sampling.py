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
max_budget=0.1
weight=1
epochs=256
a=np.sqrt(weight)
item_privacy=f"{a} {a} {a}"
user_privacy=f"{a} {a} {a}"
for lr in [0.0001]:
    for embedding_dim in [10]:
        filename=f"./Results/sampling/sampling_lr={lr}_dim={embedding_dim}_maxbudget={max_budget}_weight={weight}_epochs={epochs}.csv"
        logfile=f"./log/sampling/sampling_lr={lr}_dim={embedding_dim}_maxbudget={max_budget}_weight={weight}_epochs={epochs}.log"

        str2=f""" python mf_sampling_decentralized.py --learning_rate {lr} --epochs {epochs} --embedding_dim {embedding_dim} --max_budget {max_budget} --filename "{filename}" --logfile "{logfile}" --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" """ 

        with open('run_sampling_decentralized.sh','w') as f:   
            f.write(str1+str2)

        #run .sh file
        cmd = 'bsub < run_sampling_decentralized.sh'
        os.system(cmd)
        time.sleep(10)
