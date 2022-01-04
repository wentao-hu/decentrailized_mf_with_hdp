'''
author: wentao hu (stevenhwt@gmail.com)
input: some folder that stores the cross-validaiton results
output: the best hyperparameter combination in this folder
'''

import os
import re
import pandas as pd

def get_filelist(folder,pattern):
    filelist=[]
    for filename in os.listdir(folder):
        if re.search(pattern,filename):
            filelist.append(filename)
    return filelist
        

def read_target_line(filename,num):
    '''read target line from text file'''
    with open(filename,"r") as file:
        line=file.readline()
        counts=1
        while line:
            if counts==num:
                break
            line=file.readline()
            counts+=1
    return line


def main():
    data="ml-100k"
    method="sampling"
    dim=10
    mse_dict={}
    # for frac in [0.2,0.4,0.6,0.8,1]:
    for seed in [2]:
        resultsfolder=f"results-{data}/{method}-dpmf/seed{seed}"

        #check csv results and return
        best_result=[]
        # print(f"\nThe results of {method} on {data}:,random seed={seed} ")
        uc_range=[0.1,0.2,0.3,0.4]
        
        for uc in uc_range:
            pattern=f"epsilon_uc{uc}_{method}_test_dim={dim}.*.csv"
            filelist=get_filelist(resultsfolder,pattern)
            filelist=sorted(filelist)
            
            for file in filelist:
                filename=resultsfolder+"/"+file
                line=read_target_line(filename,101)
                avg_mse=float(line.split(",")[-1])
                # file=f"frac{frac}_{file}"
                mse_dict[file]=avg_mse
            # min_key=min(mse_dict,key=mse_dict.get)
            # best_result.append([min_key,mse_dict[min_key]])
            # print("====best hyperparameter:",min_key,mse_dict[min_key],"====\n")
    mse_dict=dict(sorted(mse_dict.items()))
    for k in sorted(mse_dict.keys()):
        print(k,mse_dict[k])
        #wirte the best hyperparameter and corresponding results into .csv file
    df=pd.DataFrame(mse_dict.items())
    df.to_csv(f"./summary/{method}_on_{data}_dim={dim}_results.csv")

if __name__=="__main__":
    main()