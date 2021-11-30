
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log-ml-1m/clusterlog/sampling/%J.err 
#BSUB -o ./log-ml-1m/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --data "Data/ml-1m" --mode "test" --regularization 0.01 --user_privacy "0.5 0.75 1" --item_privacy "0.5 0.75 1" --lr_scheme "30 50" --embedding_dim 3 --filename "./Results-ml-1m/sampling/priv1_sampling_test_dim=3_lrs=30 50_reg=0.01.csv" --logfile "./log-ml-1m/sampling/priv1_sampling_test_dim=3_lrs=30 50_reg=0.01.log" 