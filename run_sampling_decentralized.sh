
#/bin/bash
#BSUB -J sampling 
#BSUB -e ./log-ml-1m/log/%J.err 
#BSUB -o ./log-ml-1m/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --data "Data/ml-1m" --user_ratio "0.4 0.37 0.22999999999999998" --strategy min --mode "cv" --lr 0.0001 --embedding_dim 5 --regularization 0.01 --filename "results-ml-1m/sampling-dpmf/seed2/f_uc0.4_sampling_cv_dim=5_lr=0.0001_reg=0.01_seed2.csv" --logfile "log-ml-1m/sampling-dpmf/seed2/f_uc0.4_sampling_cv_dim=5_lr=0.0001_reg=0.01_seed2.log" 