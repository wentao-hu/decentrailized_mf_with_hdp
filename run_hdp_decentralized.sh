
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/hdp/%J.err 
#BSUB -o ./log/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

 python mf_hdp_decentralized.py --data "Data/ml-100k" --mode "cv" --regularization 0.01 --user_privacy "0.1 0.2 1" --item_privacy "0.1 0.2 1" --lr_scheme "45 50" --embedding_dim 2 --filename "./Results/hdp/priv1_hdp_cv_dim=2_lrs=45 50_reg=0.01.csv" --logfile "./log/hdp/priv1_hdp_cv_dim=2_lrs=45 50_reg=0.01.log" 