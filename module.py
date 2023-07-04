import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, output_channels, image_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, 256*image_size // 16*image_size // 16)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2)
        self.image_size = image_size

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, self.image_size // 16, self.image_size // 16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, image_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear((image_size // 8) * (image_size // 8) * 256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    

# 定义异常分数的损失函数
class PixelAnomalyLoss(nn.Module):
    def __init__(self):
        super(PixelAnomalyLoss, self).__init__()

    def forward(self, outputs, targets):
        anomaly_scores = torch.abs(outputs - targets)  # 计算异常分数，可以根据需求使用其他度量

        return anomaly_scores