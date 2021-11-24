
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/sampling/%J.err 
#BSUB -o ./log/clusterlog/sampling/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_sampling_decentralized.py --lr_scheme "20 40" --embedding_dim 10 --filename "./Results/sampling/sampling_dim=10.csv" --logfile "./log/sampling/sampling_dim=10.log" 