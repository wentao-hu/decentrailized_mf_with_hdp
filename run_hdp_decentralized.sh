
    #/bin/bash
    #BSUB -J hdp 
    #BSUB -e ./log-ml-1m-1/log/%J.err 
    #BSUB -o ./log-ml-1m-1/log/%J.out
    #BSUB -n 1
    #BSUB -q volta
    #BSUB -gpu "num=1:mode=exclusive_process"
     python mf_hdp_decentralized.py --data "Data/ml-1m-1" --user_privacy "0.1 0.5 1"  --mode "test" --lr 0.005 --embedding_dim 5 --regularization 0.01 --filename "results-ml-1m-1/hdp/seed2/epsilon_uc0.1_hdp_test_dim=5_lr=0.005_reg=0.01_seed2.csv" --logfile "log-ml-1m-1/hdp/seed2/epsilon_uc0.1_hdp_test_dim=5_lr=0.005_reg=0.01_seed2.log" 