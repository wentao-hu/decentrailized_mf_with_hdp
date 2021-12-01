import os
import time
import numpy as np


data="ml-1m"
datapath=f"Data/{data}"

#experiment environment
method="hdp"
mode="test"
priv_schemename="priv1"
priv_scheme="0.5 0.75 1"
user_privacy=priv_scheme
item_privacy=priv_scheme

#hyperparameter range
lr1_range=[20,30]
lr2_range=[40,50]
dim_range=[1,2,3,4,5]
reg_range=[0.01,0.001]


#write command
str1=f"""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-{data}/clusterlog/{method}/%J.err 
#BSUB -o ./log-{data}/clusterlog/{method}/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
"""

#创建相应文件夹存储结果，如果不存在就新建
# dirs = '/Users/joseph/work/python/'
# if not os.path.exists(dirs):
#     os.makedirs(dirs)

#未完待续，后面继续完善
if method=="hdp" or method=="sampling":
    for lr1 in lr1_range:
        for lr2 in lr2_range:
            for embedding_dim in dim_range:
                for reg in reg_range:
                    lr_scheme=f"{lr1} {lr2}"
                    filename=f"./Results-{data}/{method}/priv1_{method}_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.csv"
                    logfile=f"./log-{data}/{method}/priv1_{method}_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.log"

                    str2=f""" python mf_{method}_decentralized.py --data "{datapath}" --mode "{mode}" --regularization {reg} --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
                    with open(f'run_{method}_decentralized.sh','w') as f:   
                        f.write(str1+str2)

                    #run .sh file
                    cmd = f'bsub < run_{method}_decentralized.sh'
                    os.system(cmd)
                    time.sleep(10)