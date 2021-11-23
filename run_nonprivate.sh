
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/nonprivate/%J.err 
#BSUB -o ./log/clusterlog/nonprivate/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --embedding_dim 20 --filename "./Results/nonprivate/nonprivate_dim=20.csv" --logfile "./log/nonprivate/nonprivate_dim=20.log" 