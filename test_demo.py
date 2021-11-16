fraction=[0.54,0.37,0.09]
embedding_dim=8
epochs=128
filename=f"./Results/default/hdp_decentralized_epoch={epochs}_r.csv"
str2=f"""
python mf_hdp_decentralized.py --embedding_dim {embedding_dim} --epochs {epochs} --filename "{filename}" --fraction {fraction}
""" 
print(str2)