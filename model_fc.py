import torch.nn as nn

# 定义D
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = self.linear(input)
        return input

# G
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    def forward(self,input):
        input = self.linear(input)
        return input

