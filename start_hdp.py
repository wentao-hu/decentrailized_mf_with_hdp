import os
import time
import numpy as np

data="ml-1m"
datapath=f"Data/{data}"

str1=f"""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-{data}/clusterlog/hdp/%J.err 
#BSUB -o ./log-{data}/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
"""

mode="test"
user_privacy="0.5 0.75 1"
item_privacy="0.5 0.75 1"
for lr1 in [30]:
    for lr2 in [50]:
        for embedding_dim in [3]:
            for reg in [0.01]:
                lr_scheme=f"{lr1} {lr2}"
                filename=f"./Results-{data}/hdp/priv1_hdp_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.csv"
                logfile=f"./log-{data}/hdp/priv1_hdp_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}.log"

                str2=f""" python mf_hdp_decentralized.py --data "{datapath}" --mode "{mode}" --regularization {reg} --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
                with open('run_hdp_decentralized.sh','w') as f:   
                    f.write(str1+str2)

                #run .sh file
                cmd = 'bsub < run_hdp_decentralized.sh'
                os.system(cmd)
                time.sleep(10)
