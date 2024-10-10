import pandas as pd
from recbole.quick_start import load_data_and_model
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

# 检查并选择 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def user_embedding(data_path, model_path):
    # 读取.ent文件
    df = pd.read_csv(data_path, sep='\t')

    # 将字符串转换回NumPy数组，使用空格作为分隔符
    np_aligned_embeddings = np.array([np.fromstring(vec, sep=' ') for vec in df['usr_emb:float_seq']])

    config, model, dataset, train_data, valid_data, _ = load_data_and_model(model_file=model_path)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练后查看Embedding层的权重
    user_embedding = model.user_embedding.to(device)

    # 获取Embedding层的权重矩阵
    embedding_weights = user_embedding.weight.to(device)

    # 将aligned_embeddings转换为Tensor，并移动到相同设备
    aligned_embeddings = torch.tensor(np_aligned_embeddings, dtype=torch.float).to(device)

    # 获取第1个用户的两个向量
    embedding_vector_before = aligned_embeddings[10]
    embedding_vector_after = embedding_weights.detach()[10]

    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(embedding_vector_before.unsqueeze(0),
                                            embedding_vector_after.unsqueeze(0)).item()
    print(f'Cosine Similarity: {cosine_similarity:.4f}')

    # 计算欧氏距离
    euclidean_distance = torch.dist(embedding_vector_before, embedding_vector_after, p=2).item()
    print(f'Euclidean Distance: {euclidean_distance:.4f}')

    # 计算曼哈顿距离
    manhattan_distance = torch.dist(embedding_vector_before, embedding_vector_after, p=1).item()
    print(f'Manhattan Distance: {manhattan_distance:.4f}')

data_path = "saved/aligned_user_embeddings_amazon.ent"
model_path = '../../RecBole-GNN/saved/NCL-Sep-09-2024_13-35-30.pth'
user_embedding(data_path, model_path)

# def user_embedding(data_path, model_path):
    # # 读取.ent文件
    # df = pd.read_csv("saved/aligned_user_embeddings_amazon.ent", sep='\t')
    #
    # # 将字符串转换回NumPy数组，使用空格作为分隔符
    # np_aligned_embeddings = np.array([np.fromstring(vec, sep=' ') for vec in df['usr_emb:float_seq']])
    #
    # config, model, dataset, train_data, valid_data, _ = load_data_and_model(
    #     model_file=model_path,
    # )
    #
    # # 训练后查看Embedding层的权重
    # user_embedding = model.user_embedding
    # print(type(user_embedding)) # <class 'torch.nn.modules.sparse.Embedding'>
    # # 获取Embedding层的权重矩阵
    # embedding_weights = user_embedding.weight
    # print(type(embedding_weights))# <class 'torch.nn.parameter.Parameter'>
    # print(type(embedding_weights.detach())) # <class 'torch.Tensor'>
    # print(embedding_weights.detach().shape) # torch.Size([11001, 64])

    # user_emb = pd.read_pickle(data_path + '/usr_emb_np.pkl')
    # print(type(user_emb)) # <class 'numpy.ndarray'>
    # print(user_emb.shape) #(11000, 1536)
    # config, model, dataset, train_data, valid_data, _ = load_data_and_model(
    #     model_file=model_path,
    # )
    #
    # # 训练后查看Embedding层的权重
    # user_embedding = model.user_embedding
    # print(type(user_embedding)) # <class 'torch.nn.modules.sparse.Embedding'>
    # # 获取Embedding层的权重矩阵
    # embedding_weights = user_embedding.weight
    # print(type(embedding_weights))# <class 'torch.nn.parameter.Parameter'>
    # print(type(embedding_weights.detach())) # <class 'torch.Tensor'>
    # print(embedding_weights.detach().shape) # torch.Size([11001, 64])




