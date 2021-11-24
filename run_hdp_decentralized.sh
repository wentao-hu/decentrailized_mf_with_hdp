
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/hdp/%J.err 
#BSUB -o ./log/clusterlog/hdp/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

 python mf_hdp_decentralized.py --lr_scheme "20 40" --embedding_dim 10 --filename "./Results/hdp/hdp_dim=10.csv" --logfile "./log/hdp/hdp_dim=10.log" 