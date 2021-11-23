
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/sampling/%J.err 
#BSUB -o ./log/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --learning_rate 0.0001 --epochs 256 --embedding_dim 10 --max_budget 0.1 --filename "./Results/sampling/sampling_lr=0.0001_dim=10_maxbudget=0.1_weight=1_epochs=256.csv" --logfile "./log/sampling/sampling_lr=0.0001_dim=10_maxbudget=0.1_weight=1_epochs=256.log" --user_privacy "1.0 1.0 1.0" --item_privacy "1.0 1.0 1.0" 