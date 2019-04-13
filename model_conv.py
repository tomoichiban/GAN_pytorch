import torch.nn as nn

# 定义D
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(1296, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.conv(input)
        print(input.shape)  # 1,16,9,9
        # input = input.view(input.size(0), -1)
        # print(input.shape) # 1296
        # input = self.linear(input)
        return input


# G
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.ss = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=9, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
    def forward(self,input):
        input = input.view(input.size(0), 16, 1, 1)
        input = self.ss(input)
#         print(input.shape)
        return input