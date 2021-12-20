
#/bin/bash
#BSUB -J nonprivate 
#BSUB -e ./log-ml-1m/log/%J.err 
#BSUB -o ./log-ml-1m/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --data "Data/ml-1m" --mode "cv"  --lr 0.001 --embedding_dim 5 --regularization 0.001 --filename "./results-ml-1m/nonprivate/nonprivate_cv_dim=5_lr=0.001_reg=0.001seed0.csv" --logfile "./log-ml-1m/nonprivate/nonprivate_cv_dim=5_lr=0.001_reg=0.001seed0.log" 