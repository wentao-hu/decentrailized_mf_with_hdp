
#/bin/bash
#BSUB -J hdp  
#BSUB -e ./log/clusterlog/%J.err 
#BSUB -o ./log/clusterlog/%J.out
#BSUB -n 1
#BSUB -q cauchy

        python mf_hdp_decentralized.py --learning_rate 0.1 --epochs 512 --filename ./Results/default/hdp_decentralized_lr=0.1_epochs=512.csv --logfile ./log/hdp_lr=0.1_epochs=512.log
        