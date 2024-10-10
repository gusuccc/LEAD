import torch
import torch.nn as nn

# 【判别器】
class Discriminator(nn.Module):

    def __init__(self, input_size=(5237, 1536)):
        super().__init__()

        self.model = nn.Sequential(
            # 判断一个128维?1682维度吧的向量是不是后生成的
            nn.Linear(input_size[0] + input_size[1], 600),  # 1682是vector大小，1682是CGAN的扩展维度,condition=self，GS-RS使用了自增强，所以条件向量和输入向量是一样的。我们不用自增强，所以维度是辅助信息的向量维度
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.02),

            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, vector_tensor, label_tensor):
        inputs = torch.cat((vector_tensor, label_tensor))
        return self.model(inputs)


# 【生成器】
class Generator(nn.Module):

    def __init__(self, input_size=(128, 1536), output_size=5237):
        super().__init__()

        self.model = nn.Sequential(
            # 拿到一个128+1536维的加了白噪声的随机初始化向量，生成一个9332维的向量
            nn.Linear(input_size[0] + input_size[1], 300),  # 128是随机种子的维度，1536是CGAN的扩展维度(也就是附加信息的维度)
            nn.LayerNorm(300),  # channel 方向做归一化
            nn.LeakyReLU(0.02),

            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, output_size),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)
