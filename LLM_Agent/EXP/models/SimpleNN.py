import torch
import torch.nn as nn
import torch.optim as optim
# 搭建神经网络
class SimpleNN(nn.Module):
    def __init__(self,class_number):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, class_number),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    Net = SimpleNN(10)
    print(Net)