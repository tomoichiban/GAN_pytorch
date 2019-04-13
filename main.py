import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

sys.path.append('.')
from model_fc import *
from util import *

batch_size = 64

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_data = dset.MNIST('./mnist_data', train=True, download=False, transform=transform);
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)


input_size = 10

G = Generator(input_size).to(device)
D = Discriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.99))
optim_D = torch.optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.99))

loss = nn.BCELoss()

print('Start training:... \n')
# 训练
for epoch in range(30):
    for i, data in enumerate(train_loader, 0):
        img, _ = data
        m = img.size(0)

        # 0和1标签
        label_one = torch.ones(m, 1, device=device)
        label_zero = torch.zeros(m, 1, device=device)
        real_imgs = img.view(m, -1).to(device)

        # 梯度清零
        optim_D.zero_grad()

        # 识别真图片
        real_output = D(real_imgs)
        loss_real = loss(real_output, label_one)  # 损失函数a

        # 识别生成的图片
        noise = torch.randn(m, input_size, device=device)  # 产生噪声
        fake_imgs = G(noise)  # 噪声丢给G获得图片
        fake_output = D(fake_imgs)  # 给D判断真假
        loss_fake = loss(fake_output, label_zero)  # 损失函数b

        # 梯度下降
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optim_D.step()

        # 开始训练G
        optim_G.zero_grad()
        # 获取一个跟x一样的噪声图片集合
        noise = torch.randn(m, input_size, device=device)
        # 噪声丢给G获得图片
        fake_imgs = G(noise)
        # 把刚才生成的fake_img丢进D里
        output_label = D(fake_imgs)
        # 让D判断的
        loss_G = loss(output_label, label_one)
        loss_G.backward()

        optim_G.step()

        # 每迭代50批batch就保存一次生成结果
        if i % 100 == 0:
            # loss_D = loss_real + loss_fake
            # print('Loss D:', loss_D.item(), 'Loss G:', loss_G.item())
            fake_imgs = to_img(fake_imgs)
            vutils.save_image(fake_imgs, './gan_save/fake_epoch-%s-i-%s.png' % (epoch, int(i / 50 + 1)))

    print('epoch:', epoch + 1, 'finish')

print('finish')