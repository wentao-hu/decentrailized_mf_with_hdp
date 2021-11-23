import os
import time
import numpy as np
#define bash command and write into .sh file
str1="""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/hdp/%J.err 
#BSUB -o ./log/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q cauchy
#

"""

#self-defined command for hyperparameter tuning 
max_budget=0.1
weight=1
epochs=120
a=np.sqrt(weight)
item_privacy=f"{a} {a} {a}"
user_privacy=f"{a} {a} {a}"
for lr in [0.0001]:
    for embedding_dim in [10]:
        filename=f"./Results/hdp/hdp_lr={lr}_dim={embedding_dim}_maxbudget={max_budget}_weight={weight}_epochs={epochs}_lower0.csv"
        logfile=f"./log/hdp/hdp_lr={lr}_dim={embedding_dim}_maxbudget={max_budget}_weight={weight}_epochs={epochs}_lower0.log"

        str2=f""" python mf_hdp_decentralized.py --epochs {epochs} --learning_rate {lr} --embedding_dim {embedding_dim} --max_budget {max_budget} --filename "{filename}" --logfile "{logfile}" --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" """ 

        with open('run_hdp_decentralized.sh','w') as f:   
            f.write(str1+str2)

        #run .sh file
        cmd = 'bsub < run_hdp_decentralized.sh'
        os.system(cmd)
        time.sleep(10)
