import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import random
from scipy.sparse import csr_matrix

def generate_random_vector(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


def generate_random_one_hot(size):
    label_tensor = torch.zeros((size))
    random_idx = random.randint(0, size - 1)
    label_tensor[random_idx] = 1
    return label_tensor


def freeze(layer):
    # w1 = tf.stop_gradient(w1)
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def minmaxscaler(data):
    min = np.amin(data)
    max = np.amax(data)
    return (data - min) / (max - min)


# 初始化一个长度为1682的零向量 q。这个向量可能代表了某种得分或评级，长度1682可能对应了某个特定数据集中的项目数量。
# 循环三次（for i in range(3)），每次循环执行以下操作：
# 计算 array[i]（代表第 i 个虚拟邻居的评分向量）与给定的 vector（可能是当前用户的评分向量）之间的余弦相似度。
# 余弦相似度是一个衡量两个向量在方向上相似程度的度量，其值范围从-1到1，其中1表示完全相似，-1表示完全相反，0表示没有相似性。
# 打印出计算得到的余弦相似度。
# 将计算得到的相似度乘以 array[i]（邻居的评分向量），并累加到 q 向量上。
# 循环结束后，q 向量中包含了与 vector 相似度加权后的所有虚拟邻居的评分。
# 使用 minmaxscaler 函数对 q 向量进行归一化处理，将所有的评分缩放到一个固定的范围内（通常是0到1）。
# 归一化是常用的数据预处理步骤，有助于改善不同量级数据带来的影响。
def conto1(array, vector, item_num = 9332):
    # 这样，conto1 函数的输出就是一个考虑了当前用户与虚拟邻居相似度的加权评分向量，
    # 经过归一化处理后的结果。这个结果可以用于推荐系统中，作为用户可能感兴趣的项目的一个评分指标。
    q = np.zeros(item_num)
    for i in range(3):
        cosv = cosine_similarity(array[i].reshape(1, -1), vector.reshape(1, -1))
        # print(cosv)
        q = q + cosv * array[i]
    return minmaxscaler(q)


def change_date(a, epoch, path):
    b = []
    for i in tqdm(range(a.shape[0]), desc="Processing Rows"):
        for j in range(a.shape[1]):
            if (~np.isnan(a[i][j])):
                c = []
                c.append(i) # uid
                c.append(j) # iid
                c.append(a[i][j]) # rating
                # c.append(random.randint(879000000, 892999999)) # timestamp
                b.append(c)
    df = pd.DataFrame(b)
    df.to_csv(path + '/data_df' + str(epoch) + '.csv', index=False, header=False)


