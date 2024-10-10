import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from tqdm import tqdm
from cgan.model import Generator, Discriminator
from cgan.utils import generate_random_seed, minmaxscaler, conto1, change_date

parser = argparse.ArgumentParser(description='')
parser.add_argument('--x_size', type=int, default=5237, help='item number') #9332 11010 5237
parser.add_argument('--y_size', type=int, default=1536, help='condition length')#1536
parser.add_argument('--z_size', type=int, default=128, help='batch size')#128
parser.add_argument('--w_size', type=int, default=23310, help='user number')#11000 11091 23310
parser.add_argument('--num_neighbors', type=int, default=5, help='virtual neighbor number')#3
parser.add_argument('--threshold_fun', type=float, default=1, help='')#0.9 0.95 0.97
parser.add_argument('--threshold_sat', type=float, default=0.5, help='')#0.5
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')#10
parser.add_argument('--lr', type=float, default=1e-3, help='adam:learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
# parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')




if __name__ == '__main__':
    opt = parser.parse_args()
    X_SIZE = opt.x_size
    Y_SIZE = opt.y_size
    Z_SIZE = opt.z_size
    SEED = opt.seed
    EPOCH = opt.num_epochs
    dataset_name = "steam"
    # PATIENCE = opt.patience

    # 设置神经网络参数随机初始化种子，使每次训练初始参数可控
    torch.cuda.manual_seed(SEED)

    # model和rating保存路径
    g_save_path = 'saved_gan_model/{}-{}-{}/Generator'.format(dataset_name,
                                                              opt.num_neighbors,
                                                              opt.threshold_fun)
    r_save_path = 'saved_gan_model/{}-{}-{}/Rating'.format(dataset_name,
                                                           opt.num_neighbors,
                                                           opt.threshold_fun)
    os.makedirs(g_save_path, exist_ok=True)
    os.makedirs(r_save_path, exist_ok=True)

    # rating矩阵文件存放路径
    file_path = '../data/steam'

    # 读取数据
    fun_matrix = np.load('{}/fun_matrix.npy'.format(file_path))  #amazon_book 11000*9332 #yelp 11091*11010 # steam 23310*5237
    user_vectors = pd.read_pickle('{}/usr_emb_np.pkl'.format(file_path)) # 11000*1536

    raw_rating = np.load('{}/raw_fun.npy'.format(file_path))
    raw_fun = np.load('{}/raw_fun.npy'.format(file_path))

    real_rating = raw_rating.copy()
    # 使用np.isnan函数检查raw_rating数组中的缺失值（NaN），并将这些缺失值替换为0。
    raw_rating[np.isnan(raw_rating)] = 0
    raw_fun[np.isnan(raw_fun)] = 0

    # 实例化模型类
    fun_netG = Generator(input_size=(Z_SIZE, Y_SIZE), output_size=X_SIZE)
    fun_netD = Discriminator(input_size=(X_SIZE, Y_SIZE))
    # sat_netG = Generator(input_size=(Z_SIZE, Y_SIZE), output_size=X_SIZE)
    # sat_netD = Discriminator(input_size=(X_SIZE, Y_SIZE))

    # 定义损失函数类型和Optimizer
    criterion = nn.BCELoss()
    optim_G = torch.optim.SGD(fun_netG.parameters(), lr=opt.lr)
    optim_D = torch.optim.SGD(fun_netD.parameters(), lr=opt.lr)
    # optim_sG = torch.optim.SGD(sat_netG.parameters(), lr=opt.lr, momentum=opt.momentum)
    # optim_sD = torch.optim.SGD(sat_netD.parameters(), lr=opt.lr, momentum=opt.momentum)


    # 训练
    for epoch in range(1, EPOCH+1):
        print('epoch = {}'.format(epoch))
        running_loss_d = []
        running_loss_g = []
        for i in tqdm(range(fun_matrix.shape[0]), desc=f"Training Epoch {epoch+1}/{EPOCH}"):
            # 取出兴趣矩阵和满意度矩阵的一行
            user_interaction = torch.from_numpy(fun_matrix[i].astype(np.float32))
            # print(user_interaction.shape)
            user_vector = torch.from_numpy(user_vectors[i].astype(np.float32))
            # print(user_vector.shape)

            # 训练判别器
            real_output = fun_netD(user_interaction, user_vector)
            noise = generate_random_seed(Z_SIZE) # 生成噪声
            fake_interaction = fun_netG(noise, user_vector)  # 生成假样本
            fake_output = fun_netD(fake_interaction.detach(), user_vector)  # 不更新判别器的梯度

            # 创建真实和虚假标签
            target_real = torch.ones(real_output.size())
            target_fake = torch.zeros(fake_output.size())

            # 计算损失
            d_loss = criterion(real_output, target_real) + criterion(fake_output, target_fake)

            # 优化判别器
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # 训练生成器
            fake_output = fun_netD(fake_interaction, user_vector)  # 重新计算生成器输出
            g_loss = criterion(fake_output, target_real)  # 目标为1

            # 优化生成器
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            # if i % 1000 == 0:
            #     print(f'Epoch [{epoch}/{EPOCH}], Step [{i}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

            running_loss_d.append(d_loss.item())
            running_loss_g.append(g_loss.item())

        # 计算平均损失
        d_val_loss = np.mean(running_loss_d)  # 可以根据需要选择验证损失的计算方式
        g_val_loss = np.mean(running_loss_g)  # 可以根据需要选择验证损失的计算方式
        print(f'Epoch [{epoch}/{EPOCH}], D Loss: {d_val_loss.item()}, G Loss: {g_val_loss.item()}')
        # 保存每个轮次的模型
        torch.save(fun_netG.state_dict(), g_save_path + '/final_model_epoch_{}.pth'.format(epoch))

    # 保存训练数据
    # torch.save(fun_netG.state_dict(), g_save_path + '/final_model_{}'.format(epoch))
    # torch.save(sat_netG.state_dict(), g_save_path + '/sat/final_model_{}'.format(epoch))
        def saveRating(epoch):
            epoch_rating = real_rating.copy()
            for i in tqdm(range(opt.w_size), desc="Processing Users"):
                fv = user_vectors[i].astype(np.float32)
                fv = torch.from_numpy(fv)
                #print("fv:",fv.shape) #fv: torch.Size([9332])
                # sv = sat_matrix[i].astype(np.float32)
                # sv = torch.from_numpy(sv)
                fun_vs = []
                # sat_vs = []
                # 对于每个用户（或实体），它生成指定数量的虚拟邻居。
                for j in range(opt.num_neighbors):  # neighbor_num
                    # 生成虚拟邻居的过程是通过调用 fun_netG 和 sat_netG 这两个生成器网络，并传入一个随机种子和当前用户的特征向量（fv 和 sv）来完成的。

                    fun_vs.append(fun_netG(generate_random_seed(Z_SIZE), fv).detach().numpy())
                    # sat_vs.append(sat_netG(generate_random_seed(128), sv).detach().numpy())
                    # 生成的结果被添加到 fun_vs 和 sat_vs 这两个列表中
                # 这些生成的评分随后被用于计算和更新当前用户的评分，具体是通过 conto1 函数来完成的。
                f1 = conto1(fun_vs, raw_fun[i],X_SIZE)
                # s1 = conto1(sat_vs, minmaxscaler(raw_rating[i]))
                # count = 0
                for j in range(opt.x_size):
                    # if i ==0 and f1[0][j] >=1:
                    #     count += 1
                    if f1[0][j] >= opt.threshold_fun and np.isnan(epoch_rating[i][j]):
                        epoch_rating[i][j] = 1
                # if i ==0:
                #     print("count:",count)

            change_date(epoch_rating, epoch, r_save_path)
            print("change data ",epoch)
        saveRating(epoch)
