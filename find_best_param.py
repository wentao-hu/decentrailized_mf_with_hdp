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
    data="ml-1m"
    method="hdp"
    seed=0
    resultsfolder=f"results-{data}/{method}/seed{seed}"

    #check csv results and return
    best_result=[]
    print(f"The results of {method} on {data}:\n")
    uc_range=[0.1,0.2,0.3,0.4]
    for uc in uc_range:
        pattern=f"epsilon_uc{uc}_{method}_cv_dim=5.*.csv"
        filelist=get_filelist(resultsfolder,pattern)
        filelist=sorted(filelist)
        mse_dict={}
        for file in filelist:
            filename=resultsfolder+"/"+file
            line=read_target_line(filename,101)
            avg_mse=float(line.split(",")[-1])
            mse_dict[file]=avg_mse
        min_key=min(mse_dict,key=mse_dict.get)
        best_result.append([min_key,mse_dict[min_key]])
        for k,v in mse_dict.items():
            print(k,v)
        print("====best hyperparameter:",min_key,mse_dict[min_key],"====\n")

    #wirte the best hyperparameter and corresponding results into .csv file
    df=pd.DataFrame(best_result)
    #df.to_csv(f"./summary/{method}_on_{data}_best_results.csv")

if __name__=="__main__":
    main()