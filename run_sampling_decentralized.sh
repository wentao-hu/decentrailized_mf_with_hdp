
        #/bin/bash
        #BSUB -J sampling 
        #BSUB -e ./log-ml-100k-1/log/%J.err 
        #BSUB -o ./log-ml-100k-1/log/%J.out
        #BSUB -n 1
        #BSUB -q volta
        #BSUB -gpu "num=1:mode=exclusive_process"
         python mf_sampling_decentralized.py --data "Data/ml-100k-1" --user_privacy "0.1 0.5 1"  --mode "test" --lr 0.005 --embedding_dim 10 --regularization 0.01 --filename "results-ml-100k-1/sampling/seed0/epsilon_uc0.1_sampling_test_dim=10_lr=0.005_reg=0.01_seed0.csv" --logfile "log-ml-100k-1/sampling/seed0/epsilon_uc0.1_sampling_test_dim=10_lr=0.005_reg=0.01_seed0.log" 