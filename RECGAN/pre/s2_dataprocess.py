import pandas as pd
import numpy as np
from tqdm import tqdm  # 导入 tqdm


def generate_interaction(path):
    # 读取评分数据
    ratings = pd.read_csv('{}/interactions.csv'.format(path), header=None)
    n_users = ratings[0].unique().shape[0]  # 11000
    print("n_users:", n_users)
    n_items = ratings[1].unique().shape[0]  # 9332
    print("n_items:", n_items)

    # 创建初始化的稀疏矩阵
    fun_matrix = np.full((n_users, n_items), np.nan)

    # 填充 fun_matrix
    for line in ratings.itertuples():
        fun_matrix[line[1], line[2]] = line[3]

    np.save("{}/raw_fun.npy".format(path), fun_matrix)

    # 定义生成高斯白噪声的函数
    def wgn(x, snr):
        batch_size, len_x = x.shape
        Ps = np.sum(np.power(x, 2)) / len_x
        Pn = Ps / (np.power(10, snr / 10))
        noise = np.random.randn(len_x) * np.sqrt(Pn)
        return x + noise

    # 使用 tqdm 包装循环
    for i in tqdm(range(n_users), desc="Processing Users"):
        FT = np.count_nonzero(np.isnan(fun_matrix[i]))  # 计算缺失值的数量
        fx = np.zeros((1, FT)) + 0.5  # 初始化 fx
        fx_noise = wgn(fx, 50)  # 生成噪声
        fk = 0  # 用于 fx_noise 的索引

        # 遍历每个项目，填充缺失值
        for j in range(n_items):
            if np.isnan(fun_matrix[i][j]):
                fun_matrix[i][j] = fx_noise[0][fk]
                fk += 1

    np.save("{}/fun_matrix.npy".format(path), fun_matrix)

generate_interaction("../data/steam")
generate_interaction("../data/yelp")
# amazon
# yelp
# steam