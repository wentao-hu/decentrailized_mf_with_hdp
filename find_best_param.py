'''
author: wentao hu (stevenhwt@gmail.com)
input: some folder that stores the cross-validaiton results
output: the best hyperparameter combination in this folder
'''

import os
import re



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
    method="nonprivate"
    resultsfolder=f"results-{data}/{method}"

    #check csv results and return
    pattern=f"{method}_cv.*"
    filelist=get_filelist(resultsfolder,pattern)
    print(filelist)
    mse_dict={}
    for file in filelist:
        filename=resultsfolder+"/"+file
        line=read_target_line(filename,61)
        avg_mse=float(line.split(",")[-1])
        mse_dict[file]=avg_mse
    min_key=min(mse_dict,key=mse_dict.get)
    print("\nThe best hyperparameter:",min_key,mse_dict[min_key])

if __name__=="__main__":
    main()