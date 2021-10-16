# Decentralized MF with HDP 

This is our implementation for "dencentralized matrix factorization with heterogeneous differential privacy":

The code for basic matrix factorization is based the following paper and their github repository [dot_vs_learned_similarity](https://github.com/google-research/google-research/tree/master/dot_vs_learned_similarity)

*Rendle, Steffen and Krichene, Walid and Zhang, Li and Anderson, John* (2020). [Neural Collaborative Filtering vs. Matrix Factorization Revisited.](https://dl.acm.org/doi/10.1145/3383313.3412488) In Proceedings of RecSys '20.

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run HDP_centralized, the following is shown in **run_hdp_centralized.sh**:
```
python mf_simple_hdp_centralized.py --data Data/ml-1m --max_budget 1 --nonprivate_epochs 54 --private_epochs 54 --embedding_dim 8 --regularization 0.005 --learning_rate 0.002 --stddev 0.1

```

Run Sampling_centralized, the following is shown in **run_sampling_centralized.sh**:
```
python mf_simple_sampling_centralized.py --data Data/ml-1m --max_budget 1 --threshold 1 --nonprivate_epochs 54 --private_epochs 54 --embedding_dim 8 --regularization 0.005 --learning_rate 0.002 --stddev 0.1
```




### Dataset
We provide two processed datasets: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20). 

train.rating: 
- Train file.
- Each Line is a training instance: userID  itemID  rating timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID   itemID rating timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)  negativeItemID1  negativeItemID2 ...
