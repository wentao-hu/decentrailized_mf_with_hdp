import os
import time
#define bash command and write into .sh file
str1="""
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/%J.err 
#BSUB -o ./log/clusterlog/%J.out
#BSUB -n 1
#BSUB -q cauchy
"""

#self-defined command for hyperparameter tuning 
for lr in [0.01,0.05,0.1]:
    for epochs in [128,256,512]:
        filename=f"./Results/default/hdp_decentralized_lr={lr}_epochs={epochs}.csv"
        logfile=f"./log/hdp_lr={lr}_epochs={epochs}.log"

        str2=f"""
        python mf_hdp_decentralized.py --learning_rate {lr} --epochs {epochs} --filename {filename} --logfile {logfile}
        """ 

        with open('run_hdp_decentralized.sh','w') as f:   
            f.write(str1+str2)

        #run .sh file
        cmd = 'bsub < run_hdp_decentralized.sh'
        os.system(cmd)
        time.sleep(65)
