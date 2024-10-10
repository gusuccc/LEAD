import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from recbole.quick_start import load_data_and_model
import numpy as np

def run(data_path, model_path, user_num, dataset_name):
    # 检查并选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取预训练的用户嵌入
    user_emb = pd.read_pickle(data_path + '/usr_emb_np.pkl')
    print(type(user_emb))  # <class 'numpy.ndarray'>
    print(user_emb.shape)  # (11000, 1536),(11091, 11010),(23310,5238)

    # 加载模型和数据
    config, model, dataset, train_data, valid_data, _ = load_data_and_model(
        model_file=model_path,
    )

    # 获取模型的用户嵌入
    user_embedding = model.user_embedding
    embedding_weights = user_embedding.weight.detach().to(device)
    print(type(embedding_weights))  # <class 'torch.nn.parameter.Parameter'>
    print(embedding_weights.shape)  # torch.Size([11001, 64])

    # 保存最后一行用户嵌入
    last_user_embedding = embedding_weights[-1, :]

    # 丢弃推荐系统用户嵌入的最后一行
    recommender_user_embeddings = embedding_weights[:user_num, :]

    # 将预训练的用户嵌入转换为tensor并移动到设备上
    pretrained_user_embeddings = torch.tensor(user_emb, dtype=torch.float32).to(device)

    # 定义对比学习模型和数据集
    contrastive_model = ContrastiveModel().to(device)
    dataset = UserDataset(pretrained_user_embeddings, recommender_user_embeddings)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 训练模型
    optimizer = optim.Adam(contrastive_model.parameters(), lr=0.001)
    num_epochs = 500  # 最大训练轮数
    patience = 5  # 早停等待的轮数
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0  # 记录每个epoch的总损失
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            pretrained_batch, recommender_batch = batch
            pretrained_batch = pretrained_batch.to(device)
            recommender_batch = recommender_batch.to(device)
            optimizer.zero_grad()
            outputs = contrastive_model(pretrained_batch)
            loss = contrastive_loss(recommender_batch, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 计算并显示每个epoch的平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # 对齐嵌入
    aligned_embeddings = contrastive_model(pretrained_user_embeddings).detach().cpu()

    # 将最后一行用户嵌入重新附加到对齐后的嵌入中
    last_user_embedding = last_user_embedding.unsqueeze(0).cpu()
    aligned_embeddings = torch.cat([aligned_embeddings, last_user_embedding], dim=0)

    # 保存对齐后的嵌入
    user_ids = np.arange(user_num + 1)  # 示例用户ID

    # 将张量转换为NumPy数组
    np_aligned_embeddings = aligned_embeddings.numpy()

    # 将每个嵌入向量转换为字符串，并用空格分隔
    embeddings_as_str = [' '.join(map(str, vec)) for vec in np_aligned_embeddings]

    # 创建一个DataFrame
    df = pd.DataFrame({'uid:token': user_ids, 'usr_emb:float_seq': embeddings_as_str})

    # 保存为.ent文件
    df.to_csv("saved/aligned_user_embeddings_{}.ent".format(dataset_name), sep='\t', index=False)

    # 保存训练好的对比模型
    torch.save(contrastive_model.state_dict(), "saved/contrastive_model_{}.pth".format(dataset_name))


# 定义对比学习模型
class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.fc = nn.Linear(1536, 64)

    def forward(self, x):
        return self.fc(x)


# 对比学习损失函数
def contrastive_loss(y_true, y_pred):
    batch_size = y_true.size(0)
    y_pred = y_pred / y_pred.norm(dim=1, keepdim=True)
    y_true = y_true / y_true.norm(dim=1, keepdim=True)

    pos_sim = torch.sum(y_true * y_pred, dim=-1)
    pos_sim = pos_sim.view(batch_size, 1)

    neg_sim = torch.mm(y_true, y_pred.t())

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(y_true.device)  # 正样本对的标签为0，移动到同一设备

    return nn.CrossEntropyLoss()(logits, labels)


# 数据集和数据加载器
class UserDataset(Dataset):
    def __init__(self, pretrained_embeddings, recommender_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.recommender_embeddings = recommender_embeddings

    def __len__(self):
        return self.pretrained_embeddings.size(0)

    def __getitem__(self, idx):
        return self.pretrained_embeddings[idx], self.recommender_embeddings[idx]

# 运行主函数
# 假设你有预训练的用户编码和推荐系统的用户编码

data_path1 = '../data/amazon' # amazon,yelp,steam
# (11000, 1536),(11091, 11010),(23310,5238)
model_path1 = '../../RecBole-GNN/saved/NCL-Sep-09-2024_13-35-30.pth'
# NCL-Sep-09-2024_13-35-30.pth
# NCL-Sep-09-2024_16-21-33.pth
# NCL-Sep-09-2024_15-19-28.pth
run(data_path1, model_path1, 11000, 'amazon')

data_path2 = '../data/yelp' # amazon,yelp,steam
model_path2 = '../../RecBole-GNN/saved/NCL-Sep-09-2024_16-21-33.pth'
run(data_path2, model_path2, 11091, 'yelp')

data_path3 = '../data/steam' # amazon,yelp,steam
model_path3 = '../../RecBole-GNN/saved/NCL-Sep-09-2024_15-19-28.pth'
run(data_path3, model_path3, 23310, 'steam')
