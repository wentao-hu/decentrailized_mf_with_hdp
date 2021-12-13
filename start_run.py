'''
author: Wentao Hu(stevenhwt)
'''

import os
import time

#experiment setting
data="ml-100k"
method="nonprivate"
mode="test"

datapath=f"Data/{data}"
#hyperparameter range
lr_range=[0.01]  #initial learning rate
dim_range=[4]
reg_range=[0.01]



# create folder to store log and results
dir1=f'log-{data}/{method}'
if not os.path.exists(dir1):
    os.makedirs(dir1)

dir2=f'results-{data}/{method}'
if not os.path.exists(dir2):
    os.makedirs(dir2)


#write command
str1=f"""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-{data}/log/%J.err 
#BSUB -o ./log-{data}/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
"""


if method=="nonprivate":
    for lr in lr_range:
        for dim in dim_range:
            for reg in reg_range:
                filename=f"./results-{data}/nonprivate/nonprivate_{mode}_dim={dim}_lr={lr}_reg={reg}b.csv"
                logfile=f"./log-{data}/nonprivate/nonprivate_{mode}_dim={dim}_lr={lr}_reg={reg}b.log"

                str2=f""" python mf_nonprivate.py --data "{datapath}" --mode "{mode}"  --lr {lr} --embedding_dim {dim} --regularization {reg} --filename "{filename}" --logfile "{logfile}" """
                with open(f'run_nonprivate.sh','w') as f:   
                    f.write(str1+str2)

                #run .sh file
                cmd = f'bsub < run_nonprivate.sh'
                os.system(cmd)
                time.sleep(10)


# if method=="hdp" or method=="sampling":
#     for lr1 in lr1_range:
#         for lr2 in lr2_range:
#             for embedding_dim in dim_range:
#                 for reg in reg_range:
#                     
#                     filename=f"./Results-{data}/{method}/priv1_{method}_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.csv"
#                     logfile=f"./log-{data}/{method}/priv1_{method}_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.log"

#                     str2=f""" python mf_{method}_decentralized.py --data "{datapath}" --mode "{mode}" --regularization {reg} --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
#                     with open(f'run_{method}_decentralized.sh','w') as f:   
#                         f.write(str1+str2)

#                     #run .sh file
#                     cmd = f'bsub < run_{method}_decentralized.sh'
#                     os.system(cmd)
#                     time.sleep(10)