
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/nonprivate/%J.err 
#BSUB -o ./log/clusterlog/nonprivate/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --mode "cv" --lr_scheme "30 50" --embedding_dim 10 --filename "./Results/nonprivate/nonprivate_cv_dim=10_lrs=30 50.csv" --logfile "./log/nonprivate/nonprivate_cv_dim=10_lrs=30 50.log" 