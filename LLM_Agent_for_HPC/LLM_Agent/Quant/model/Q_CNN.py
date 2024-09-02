import torch
import torch.nn as nn
import torch.optim as optimimport
import torch.nn.functional as F

from SimpleModule.Quant.Mixed.model.Qmodule import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_blocks=2,
                 conv1_x_nbit=8, conv1_w_nbit=8, conv1_q_group_size=-1,
                 block_x_nbit=8, block_w_nbit=8, block_q_group_size=-1,
                 fc1_x_nbit=8, fc1_w_nbit=8, fc1_q_group_size=-1,
                 fc2_x_nbit=8, fc2_w_nbit=8, fc2_q_group_size=-1):
        super(Q_SimpleCNN, self).__init__()

        # 保存量化配置
        self.quant_config = {
            "conv1": {"x_nbit": conv1_x_nbit, "w_nbit": conv1_w_nbit, "q_group_size": conv1_q_group_size},
            "blocks": [],
            "fc1": {"x_nbit": fc1_x_nbit, "w_nbit": fc1_w_nbit, "q_group_size": fc1_q_group_size},
            "fc2": {"x_nbit": fc2_x_nbit, "w_nbit": fc2_w_nbit, "q_group_size": fc2_q_group_size}
        }

        # 第一层卷积的量化设置
        self.conv1 = QConv2d(3, 16, kernel_size=3, padding=1,
                             x_nbit=conv1_x_nbit, w_nbit=conv1_w_nbit, q_group_size=conv1_q_group_size)
        self.pool = nn.MaxPool2d(2, 2)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                QConv2d(16, 16, kernel_size=3, padding=1,
                        x_nbit=block_x_nbit, w_nbit=block_w_nbit, q_group_size=block_q_group_size),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.blocks.append(block)

        # 第一层全连接层的量化设置
        self.fc1 = QLinear(16 * 4 * 4, 32,
                           x_nbit=fc1_x_nbit, w_nbit=fc1_w_nbit, q_group_size=fc1_q_group_size)
        # 第二层全连接层的量化设置
        self.fc2 = QLinear(32, num_classes,
                           x_nbit=fc2_x_nbit, w_nbit=fc2_w_nbit, q_group_size=fc2_q_group_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    Net = Q_SimpleCNN(num_classes=10, num_blocks=2,
                        conv1_x_nbit=6, conv1_w_nbit=6, conv1_q_group_size=4,
                        block_x_nbit=8, block_w_nbit=8, block_q_group_size=-1,
                        fc1_x_nbit=4, fc1_w_nbit=4, fc1_q_group_size=2,
                        fc2_x_nbit=8, fc2_w_nbit=8, fc2_q_group_size=-1)

    print(Net)

