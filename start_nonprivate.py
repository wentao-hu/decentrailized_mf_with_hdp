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


for embedding_dim in [2,3,4,5,10]:
    filename=f"./Results/nonprivate/nonprivate_dim={embedding_dim}.csv"
    logfile=f"./log/nonprivate/nonprivate_dim={embedding_dim}.log"

    str2=f""" python mf_nonprivate.py --embedding_dim {embedding_dim} --filename "{filename}" --logfile "{logfile}" """
    with open('run_nonprivate.sh','w') as f:   
        f.write(str1+str2)

    #run .sh file
    cmd = 'bsub < run_nonprivate.sh'
    os.system(cmd)
    time.sleep(10)
