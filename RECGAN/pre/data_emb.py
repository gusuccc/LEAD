import pandas as pd
import numpy as np


# 其实最后处理成交互矩阵.npy保存就行
# 直接读了就能用 不转换为csv
user_emb = pd.read_pickle('../data/steam/usr_emb_np.pkl')
print(user_emb.shape)
print(type(user_emb))
print(user_emb[0])
print(type(user_emb[0]))
print(user_emb[0].shape)
# 三个数据集都是1536