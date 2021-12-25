'''
author: Wentao Hu(stevenhwt@gmail.com)
to do some addtional experiments
'''
import os
import time

random_seed=2
#experiment setting

mode="test"
#hyperparameter range
dim_range=[5]
lr_range=[0.0002]  #initial learning rate
reg_range=[0.01]

#privacy setting

# #for dpmf method
# method_for_file="dpmf"
strategy="min"  # --strategy {strategy}


# create folder to store log and results
data=f"ml-1m"
datapath=f"Data/{data}"

for method in ["sampling"]:
    dir1=f'log-{data}/{method}-dpmf/seed{random_seed}'
    if not os.path.exists(dir1):
        os.makedirs(dir1)

    dir2=f'results-{data}/{method}-dpmf/seed{random_seed}'
    if not os.path.exists(dir2):
        os.makedirs(dir2)


    #write command
    str1=f"""
    #/bin/bash
    #BSUB -J {method} 
    #BSUB -e ./log-{data}/log/%J.err 
    #BSUB -o ./log-{data}/log/%J.out
    #BSUB -n 1
    #BSUB -q volta
    #BSUB -gpu "num=1:mode=exclusive_process"
    """

    uc_range=[0.1]

    if method=="hdp" or method=="sampling":
        for lr in lr_range:
            for dim in dim_range:
                for reg in reg_range:
                    for uc in uc_range:
                        f_um=0.37
                        f_ul=1-uc-f_um
                        user_ratio=f"{uc} {f_um} {f_ul}"  #change epsilon to f
                        # user_privacy=f"{uc} 0.5 1"

                        logfile=f"{dir1}/f_uc{uc}_{method}_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.log"
                        filename=f"{dir2}/f_uc{uc}_{method}_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.csv"


                        str2=f""" python mf_{method}_decentralized.py --strategy {strategy} --data "{datapath}" --user_ratio "{user_ratio}"  --mode "{mode}" --lr {lr} --embedding_dim {dim} --regularization {reg} --filename "{filename}" --logfile "{logfile}" """
                        with open(f'run_{method}_decentralized.sh','w') as f:   
                            f.write(str1+str2)
                        
                        
                        #run .sh file
                        cmd = f'bsub < run_{method}_decentralized.sh'
                        os.system(cmd)
                        time.sleep(2)
                        
                        
                        
                        
                        
                        
    if method=="nonprivate":
        for lr in lr_range:
            for dim in dim_range:
                for reg in reg_range:
                    logfile=f"{dir1}/nonprivate_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.log"
                    filename=f"{dir2}/nonprivate_{mode}_dim={dim}_lr={lr}_reg={reg}_seed{random_seed}.csv"

                    str2=f""" python mf_nonprivate.py --data "{datapath}" --mode "{mode}"  --lr {lr} --embedding_dim {dim} --regularization {reg} --filename "{filename}" --logfile "{logfile}" """
                    with open(f'run_nonprivate.sh','w') as f:   
                        f.write(str1+str2)

                    #run .sh file
                    cmd = f'bsub < run_nonprivate.sh'
                    os.system(cmd)
                    time.sleep(2)