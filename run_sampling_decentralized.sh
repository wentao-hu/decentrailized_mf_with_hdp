
#/bin/bash
#BSUB -J sampling 
#BSUB -e ./log-ml-100k/log/%J.err 
#BSUB -o ./log-ml-100k/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --data "Data/ml-100k" --strategy "min" --user_privacy "0.1 0.5 1" --mode "cv" --lr 0.005 --embedding_dim 5 --regularization 0.001 --filename "./results-ml-100k/sampling-dpmf/epsilon_uc0.1_sampling_cv_dim=5_lr=0.005_reg=0.001b.csv" --logfile "./log-ml-100k/sampling-dpmf/epsilon_uc0.1_sampling_cv_dim=5_lr=0.005_reg=0.001b.log" 