import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from module import *
from dataloader import *


input_dim=100
output_channels=1
image_size=256
input_channels=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(input_dim, output_channels, image_size).to(device)
discriminator = Discriminator(input_channels, image_size).to(device)

learning_rate=0.001
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

num_epochs=100

aset= AnomalySet('dataset',image_size=image_size)
dataloader = DataLoader(aset,batch_size=1)

# 定义损失函数和优化器
criterion = PixelAnomalyLoss()
print_interval=1
for epoch in range(num_epochs):
    for i, real_images in enumerate(dataloader):
        batch_size = real_images.size(0)
        
        real_images = real_images.to(device)
        
        # 创建标签：所有样本都被视为正常（类别0）
        real_labels = torch.zeros(batch_size, 1).to(device)
        
        ###############################
        # 训练判别器
        ###############################
        
        # 初始化判别器梯度
        optimizer_d.zero_grad()
        
        # 生成假样本
        noise = torch.randn(batch_size, input_dim).to(device)
        fake_images = generator(noise)
        
        # 创建标签：所有生成的样本都被视为异常（类别1）
        fake_labels = torch.ones(batch_size, 1).to(device)
        
        # 在判别器上计算真实样本和生成样本的输出
        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())
        
        # 计算判别器的损失函数
        d_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
        
        # 反向传播和优化判别器
        d_loss.backward()
        optimizer_d.step()
        
        ###############################
        # 训练生成器
        ###############################
        
        # 初始化生成器梯度
        optimizer_g.zero_grad()
        
        # 生成假样本
        noise = torch.randn(batch_size, input_dim).to(device)
        fake_images = generator(noise)
        
        # 在判别器上计算生成样本的输出
        outputs = discriminator(fake_images)
        
        # 计算生成器的损失函数
        g_loss = criterion(outputs, real_labels)
        
        # 反向传播和优化生成器
        g_loss.backward()
        optimizer_g.step()
        
        ###############################
        # 打印训练进度
        ###############################
        
        if (i+1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")


