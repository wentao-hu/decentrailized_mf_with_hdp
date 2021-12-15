
#/bin/bash
#BSUB -J sampling 
#BSUB -e ./log-ml-100k/log/%J.err 
#BSUB -o ./log-ml-100k/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --data "Data/ml-100k" --user_privacy "0.4 0.5 1" --mode "test" --lr 0.005 --embedding_dim 10 --regularization 0.01 --filename "./results-ml-100k/sampling/epsilon_uc0.4_sampling_test_dim=10_lr=0.005_reg=0.01b.csv" --logfile "./log-ml-100k/sampling/epsilon_uc0.4_sampling_test_dim=10_lr=0.005_reg=0.01b.log" 