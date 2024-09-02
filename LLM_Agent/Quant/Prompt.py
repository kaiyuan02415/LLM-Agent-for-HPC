# 初始化的Prompt，参考sklearn格式
Prompt0 = """
给你提供一个支持量化的神经网络：
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
您正在帮助对SimpleCNN进行量化，训练的数据集是CIFAR-10。训练是使用PyTorch进行的。
你可以修改如下参数帮助优化量化模型：
-'x_nbits':某个层次输入的量化比特数，如conv1_x_nbit，代表了conv1的输入量化比特数为8。可选的选项是：[8, 6, 4]
-'w_nbits':某个层次权重的量化比特数，如conv1_w_nbit，代表conv1的权重量化为8比特。可选选项：[8, 6, 4]
-'q_group_size'：指定分组量化的组大小，-1表示不分组。使用此参数时一定要保证对于目前的数据集输入和网络结构而言，不会报错！
- `learning_rate`：优化器的学习率。范围：[1e-05, 1.0]，默认值：0.01，使用对数刻度。
- `batch_size`：每批输入数据的样本数量。范围：[16, 256]，默认值：64，使用对数刻度。
- `weight_decay`：L2正则化系数。范围：[1e-08, 0.1]，默认值：0.0001，使用对数刻度。
- `momentum`：用于SGD优化器的动量。范围：[0.5, 0.99]，默认值：0.9。
你会在每一次后得到训练的准确率，你提供参数的依据是：在给定运行内存（8MB）的情况下，量化后的模型达到最大的准确率
你根据torchinfo.summary来估算内存占用，其传参依据本轮的配置，CIFAR-10数据集和模型本身
请注意，你每次只提供一组配置，当我将训练结果提供给你时，你再返回一组优化后的配置。
请以JSON格式提供配置。例如：{"learning_rate": x, 
                        "batch_size": y, 
                        "weight_decay": z, 
                        "momentum": k, 
                        "quant_config":{
                            "conv1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
                            "blocks":{"x_nbit": 8, "w_nbit":8, "q_group_size": -1},
                            "fc1": {"x_nbit": 4, "w_nbit": 4, "q_group_size": -1},
                            "fc2": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1}
        }
}
注意quant_config的格式要和Q_SimpleCNN中config的格保持对应
每轮还请附上你估算的本次运行内存占用(MB)
"""

# 参考格式
Prompt_iter = """
目前的配置为：
    "learning_rate": 0.0032,
    "batch_size": 64,
    "weight_decay": 0.00001,
    "momentum": 0.95,
    "quant_config":{
        "conv1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
        "blocks":{"x_nbit": 8, "w_nbit":8, "q_group_size": -1},
        "fc1": {"x_nbit": 4, "w_nbit": 4, "q_group_size": -1},
        "fc2": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1}
}
基于此配置的结果：验证正确率： xxx
请优化并提供一组优化后的配置
"""
