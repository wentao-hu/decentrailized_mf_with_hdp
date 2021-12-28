
    #/bin/bash
    #BSUB -J hdp 
    #BSUB -e ./log-ml-100k/log/%J.err 
    #BSUB -o ./log-ml-100k/log/%J.out
    #BSUB -n 1
    #BSUB -q volta
    #BSUB -gpu "num=1:mode=exclusive_process"
     python mf_hdp_decentralized.py --data "Data/ml-100k" --user_ratio "0.4 0.37 0.22999999999999998"  --mode "test" --lr 0.005 --embedding_dim 10 --regularization 0.01 --filename "results-ml-100k/hdp/seed4/f_uc0.4_hdp_test_dim=10_lr=0.005_reg=0.01_seed4.csv" --logfile "log-ml-100k/hdp/seed4/f_uc0.4_hdp_test_dim=10_lr=0.005_reg=0.01_seed4.log" 