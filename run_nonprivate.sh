
    #/bin/bash
    #BSUB -J nonprivate 
    #BSUB -e ./log-ml-1m/log/%J.err 
    #BSUB -o ./log-ml-1m/log/%J.out
    #BSUB -n 1
    #BSUB -q volta
    #BSUB -gpu "num=1:mode=exclusive_process"
     python mf_nonprivate.py --data "Data/ml-1m" --mode "test"  --lr 0.005 --embedding_dim 5 --regularization 0.01 --filename "results-ml-1m/nonprivate/seed50/nonprivate_test_dim=5_lr=0.005_reg=0.01_seed50.csv" --logfile "log-ml-1m/nonprivate/seed50/nonprivate_test_dim=5_lr=0.005_reg=0.01_seed50.log" 