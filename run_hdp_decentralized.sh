
#/bin/bash
#BSUB -J hdp 
#BSUB -e ./log-ml-1m/log/%J.err 
#BSUB -o ./log-ml-1m/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_hdp_decentralized.py --data "Data/ml-1m" --user_privacy "0.1 0.5 1" --mode "test" --lr 0.005 --embedding_dim 10 --regularization 0.01 --filename "results-ml-1m/hdp/seed3/epsilon_uc0.1_hdp_test_dim=10_lr=0.005_reg=0.01_seed3.csv" --logfile "log-ml-1m/hdp/seed3/epsilon_uc0.1_hdp_test_dim=10_lr=0.005_reg=0.01_seed3.log" 