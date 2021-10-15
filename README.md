# Decentralized MF with HDP 

This is our implementation for "dencentralized matrix factorization with heterogeneous differential privacy":

The code for basic matrix factorization is based the following paper and their github repository [dot_vs_learned_similarity](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)

*Rendle, Steffen and Krichene, Walid and Zhang, Li and Anderson, John* (2020). [Neural Collaborative Filtering vs. Matrix Factorization Revisited.](https://dl.acm.org/doi/10.1145/3383313.3412488) In Proceedings of RecSys '20.

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run HDP_centralized:
```
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run Sampling_centralized:
```
python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run HDP_decentralized: 
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run Sampling_decentralized:
```
python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5
```



### Dataset
We provide two processed datasets: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20). 

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
