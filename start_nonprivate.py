import os
import time
import numpy as np
#define bash command and write into .sh file
str1="""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/nonprivate/%J.err 
#BSUB -o ./log/clusterlog/nonprivate/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
"""

#self-defined command for hyperparameter tuning 
mode="cv"
for lr1 in [20,30]:
    for lr2 in [40,50]:
        for embedding_dim in [5,10]:
            for reg in [0.01,0.001]:
                lr_scheme=f"{lr1} {lr2}"
                filename=f"./Results/nonprivate/nonprivate_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}_priv2.csv"
                logfile=f"./log/nonprivate/nonprivate_{mode}_dim={embedding_dim}_lrs={lr_scheme}_reg={reg}_priv2.log"

                str2=f""" python mf_nonprivate.py --mode "{mode}" --regularization {reg} --lr_scheme "{lr_scheme}" --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
                with open('run_nonprivate.sh','w') as f:   
                    f.write(str1+str2)

                #run .sh file
                cmd = 'bsub < run_nonprivate.sh'
                os.system(cmd)
                time.sleep(10)
