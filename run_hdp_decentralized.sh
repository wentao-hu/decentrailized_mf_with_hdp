
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-ml-1m/clusterlog/hdp/%J.err 
#BSUB -o ./log-ml-1m/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

 python mf_hdp_decentralized.py --mode "cv" --regularization 0.001 --user_privacy "0.5 0.75 1" --item_privacy "0.5 0.75 1" --lr_scheme "30 40" --embedding_dim 2 --filename "./Results-ml-1m/hdp/priv1_hdp_cv_dim=2_lrs=30 40_reg=0.001.csv" --logfile "./log-ml-1m/hdp/priv1_hdp_cv_dim=2_lrs=30 40_reg=0.001.log" 