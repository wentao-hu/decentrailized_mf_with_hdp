
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/hdp/%J.err 
#BSUB -o ./log/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

 python mf_hdp_decentralized.py --mode "test" --user_privacy "0.1 0.2 1" --item_privacy "0.1 0.2 1" --lr_scheme "35 50" --embedding_dim 1 --filename "./Results/hdp/hdp_test_dim=1_lrs=35 50_priv2.csv" --logfile "./log/hdp/hdp_test_dim=1_lrs=35 50_priv2.log" 