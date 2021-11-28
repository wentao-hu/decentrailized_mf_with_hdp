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
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

"""

#self-defined command for hyperparameter tuning 

mode="test"
user_privacy="0.1 0.2 1"
item_privacy="0.1 0.2 1"
for lr1 in [40]:
    for lr2 in [50]:
        for embedding_dim in [1]:
            for reg in [0.001]:
                lr_scheme=f"{lr1} {lr2}"
                filename=f"./Results/hdp/hdp_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}_priv2_ub.csv"
                logfile=f"./log/hdp/hdp_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}_priv2_ub.log"

                str2=f""" python mf_hdp_decentralized.py --mode "{mode}" --regularization {reg} --user_privacy "{user_privacy}" --item_privacy "{item_privacy}" --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
                with open('run_hdp_decentralized.sh','w') as f:   
                    f.write(str1+str2)

                #run .sh file
                cmd = 'bsub < run_hdp_decentralized.sh'
                os.system(cmd)
                time.sleep(30)
