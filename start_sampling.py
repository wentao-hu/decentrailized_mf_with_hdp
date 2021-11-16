import os

#define bash command and write into .sh file
str1="""
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/%J.err 
#BSUB -o ./log/clusterlog/%J.out
#BSUB -n 1
#BSUB -q cauchy
"""

#self-defined command for hyperparameter tuning 

#write hyperparameters into bash command
#write hyperparameters into bash command
# filename=f"./Results/default/hdp_decentralized.csv"
# logfile="./log/hdp.log"

str2=f"""
python mf_sampling_decentralized.py --learning_rate 0.05
""" 

with open('run_sampling_decentralized.sh','w') as f:   
    f.write(str1+str2)



#run .sh file
cmd = 'bsub < run_sampling_decentralized.sh'
os.system(cmd)
