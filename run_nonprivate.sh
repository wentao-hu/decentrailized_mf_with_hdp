
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/nonprivate/%J.err 
#BSUB -o ./log/clusterlog/nonprivate/%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --mode "cv" --regularization 0.01 --lr_scheme "20 40" --embedding_dim 5 --filename "./Results/nonprivate/nonprivate_cv_dim=5_lrs=20 40_reg=0.01_priv2.csv" --logfile "./log/nonprivate/nonprivate_cv_dim=5_lrs=20 40_reg=0.01_priv2.log" 