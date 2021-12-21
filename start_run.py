'''
author: Wentao Hu(stevenhwt@gmail.com)
'''

import os
import time

random_seed=1

#experiment setting
data="ml-1m"
method="hdp"
mode="test"
datapath=f"Data/{data}"
#hyperparameter range
dim_range=[10]
lr_range=[0.005]  #initial learning rate
reg_range=[0.01]

# strategy="min"


# create folder to store log and results
dir1=f'log-{data}/{method}/seed{random_seed}'
if not os.path.exists(dir1):
    os.makedirs(dir1)

dir2=f'results-{data}/{method}/seed{random_seed}'
if not os.path.exists(dir2):
    os.makedirs(dir2)


#write command
str1=f"""
#/bin/bash
#BSUB -J {method} 
#BSUB -e ./log-{data}/log/%J.err 
#BSUB -o ./log-{data}/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
"""
#privacy setting
uc_range=[0.1,0.2,0.3,0.4]

if method=="hdp" or method=="sampling":
    for lr in lr_range:
        for dim in dim_range:
            for reg in reg_range:
                for uc in uc_range:
                    # f_um=0.37
                    # f_ul=1-uc-f_um
                    # user_ratio=f"{uc} {f_um} {f_ul}"
                    user_privacy=f"{uc} 0.5 1"

                    logfile=f"{dir1}/epsilon_uc{uc}_{method}_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.log"
                    filename=f"{dir2}/epsilon_uc{uc}_{method}_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.csv"


                    str2=f""" python mf_{method}_decentralized.py --data "{datapath}" --user_privacy "{user_privacy}" --mode "{mode}" --lr {lr} --embedding_dim {dim} --regularization {reg} --filename "{filename}" --logfile "{logfile}" """
                    with open(f'run_{method}_decentralized.sh','w') as f:   
                        f.write(str1+str2)

                    #run .sh file
                    cmd = f'bsub < run_{method}_decentralized.sh'
                    os.system(cmd)
                    time.sleep(5)




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
                time.sleep(5)