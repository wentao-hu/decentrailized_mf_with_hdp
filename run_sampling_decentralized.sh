
#/bin/bash
#BSUB -J sampling
#BSUB -e ./log/clusterlog/%J.err 
#BSUB -o ./log/clusterlog/%J.out
#BSUB -n 1
#BSUB -q cauchy

python mf_sampling_decentralized.py
