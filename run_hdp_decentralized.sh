
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/hdp/%J.err 
#BSUB -o ./log/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q cauchy
#

 python mf_hdp_decentralized.py --epochs 120 --learning_rate 0.0001 --embedding_dim 10 --max_budget 0.1 --filename "./Results/hdp/hdp_lr=0.0001_dim=10_maxbudget=0.1_weight=1_epochs=120_lower0.csv" --logfile "./log/hdp/hdp_lr=0.0001_dim=10_maxbudget=0.1_weight=1_epochs=120_lower0.log" --user_privacy "1.0 1.0 1.0" --item_privacy "1.0 1.0 1.0" 