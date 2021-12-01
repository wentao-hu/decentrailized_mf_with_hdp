'''
author: wentao hu (stevenhwt@gmail.com)
input: some folder that stores the cross-validaiton results
output: the best hyperparameter combination in this folder
'''

import os
import re
from types import MethodDescriptorType




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

def check_logfiles(folder,pattern):
    filelist=get_filelist(folder,pattern)
    avgmse_dict={}
    for file in filelist:
        filename=folder+"/"+file
        sum_mse=0
        for num in [64,126,188,250,312]:
            line1=read_target_line(filename,num)
            valmse=float(line1.split("\t")[-2].split("=")[-1])
            sum_mse+=valmse
        avgmse_dict[file]=sum_mse/5
    min_key=min(avgmse_dict,key=avgmse_dict.get)
    print(min_key,avgmse_dict[min_key]) 


def main():
    data="ml-1m"
    method="hdp"
    logfolder=f"log-{data}/{method}"
    resultsfolder=f"Results-{data}/{method}"

    #check csv results and return
    pattern1=f"priv1_{method}_cv.*"
    # pattern2=f"{method}_cv.*priv2.*"
    filelist=get_filelist(logfolder,pattern1)
    print(filelist)
    avgmse_dict={}
    for file in filelist:
        filename=resultsfolder+"/"+file
        line=read_target_line(filename,7)
        avg_mse=float(line.split(",")[-1])
        avgmse_dict[file]=avg_mse
    min_key=min(avgmse_dict,key=avgmse_dict.get)
    print("\nThe best hyperparameter:",min_key,avgmse_dict[min_key])

if __name__=="__main__":
    main()