
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/sampling/%J.err 
#BSUB -o ./log/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --mode "test" --regularization 0.001 --user_privacy "0.1 0.2 1" --item_privacy "0.1 0.2 1" --lr_scheme "20 40" --embedding_dim 1 --filename "./Results/sampling/sampling_test_dim=1_lrs=20 40_reg=0.001_priv2_ub.csv" --logfile "./log/sampling/sampling_test_dim=1_lrs=20 40_reg=0.001_priv2_ub.log" 