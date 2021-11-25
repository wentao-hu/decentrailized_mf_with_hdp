
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/sampling/%J.err 
#BSUB -o ./log/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --mode "cv" --regularization 0.001 --user_privacy "0.1 0.2 1" --item_privacy "0.1 0.2 1" --lr_scheme "20 50" --embedding_dim 2 --filename "./Results/sampling/sampling_cv_dim=2_lrs=20 50_reg=0.001_priv2.csv" --logfile "./log/sampling/sampling_cv_dim=2_lrs=20 50_reg=0.001_priv2.log" 