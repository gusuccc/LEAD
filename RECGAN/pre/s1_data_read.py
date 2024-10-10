import pandas as pd
import numpy as np


def mergeset(path):
    dataset_trn = pd.read_pickle('{}/trn_mat.pkl'.format(path))
    print(dataset_trn.shape)
    dataset_tst = pd.read_pickle('{}/tst_mat.pkl'.format(path))
    dataset_val = pd.read_pickle('{}/val_mat.pkl'.format(path))
    # 合并矩阵
    combined_matrix = dataset_trn + dataset_tst + dataset_val
    print(combined_matrix.shape)
    # 获取非零元素
    non_zero_values = combined_matrix.data

    # 检查是否存在除了 0 或 1 之外的值
    if any(value not in (0, 1) for value in non_zero_values):
        print("存在除了 0 或 1 之外的值")
    else:
        print("全部值为 0 或 1")
    # 直接从稀疏矩阵提取非零元素并保存为 CSV
    rows, cols = combined_matrix.nonzero()
    data = combined_matrix.data

    # 创建 DataFrame
    df = pd.DataFrame({'uid': rows, 'iid': cols, 'inter': data})
    print(df.head())
    print(df.shape)
    # 保存为 CSV 格式
    df.to_csv('{}/interactions.csv'.format(path), index=False, header=False)

#mergeset("../data/amazon") # 11000*9332
mergeset("../data/yelp") # 11091*11010  166620inter
mergeset("../data/steam")# 23310*5237  316190inter