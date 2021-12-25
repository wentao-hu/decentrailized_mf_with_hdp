
    #/bin/bash
    #BSUB -J sampling 
    #BSUB -e ./log-ml-1m/log/%J.err 
    #BSUB -o ./log-ml-1m/log/%J.out
    #BSUB -n 1
    #BSUB -q volta
    #BSUB -gpu "num=1:mode=exclusive_process"
     python mf_sampling_decentralized.py --strategy min --data "Data/ml-1m" --user_ratio "0.1 0.37 0.53"  --mode "test" --lr 0.0002 --embedding_dim 5 --regularization 0.01 --filename "results-ml-1m/sampling-dpmf/seed2/f_uc0.1_sampling_test_dim=5_lr=0.0002_reg=0.01_seed2.csv" --logfile "log-ml-1m/sampling-dpmf/seed2/f_uc0.1_sampling_test_dim=5_lr=0.0002_reg=0.01_seed2.log" 