
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log-ml-100k/log/%J.err 
#BSUB -o ./log-ml-100k/log/%J.out
#BSUB -n 1
#BSUB -q volta
#BSUB -gpu "num=1:mode=exclusive_process"
 python mf_nonprivate.py --data "Data/ml-100k" --mode "test"  --lr 0.01 --embedding_dim 4 --regularization 0.01 --filename "./results-ml-100k/nonprivate/nonprivate_test_dim=4_lr=0.01_reg=0.01b.csv" --logfile "./log-ml-100k/nonprivate/nonprivate_test_dim=4_lr=0.01_reg=0.01b.log" 