
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-ml-1m/clusterlog/hdp/%J.err 
#BSUB -o ./log-ml-1m/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_hdp_decentralized.py --data "Data/ml-1m" --mode "test" --regularization 0.01 --user_privacy "0.5 0.75 1" --item_privacy "0.5 0.75 1" --lr_scheme "30 50" --embedding_dim 4 --filename "./Results-ml-1m/hdp/priv1_hdp_test_dim=4_lrs=30 50_reg=0.01.csv" --logfile "./log-ml-1m/hdp/priv1_hdp_test_dim=4_lrs=30 50_reg=0.01.log" 