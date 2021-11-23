
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/nonprivate/%J.err 
#BSUB -o ./log/clusterlog/nonprivate/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --epochs 256 --learning_rate 0.0001 --embedding_dim 10 --filename "./Results/nonprivate/nonprivate_lr=0.0001_dim=10.csv" --logfile "./log/nonprivate/nonprivate_lr=0.0001_dim=10.log" 