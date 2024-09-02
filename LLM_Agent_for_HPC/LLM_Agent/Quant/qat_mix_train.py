import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
from torchinfo import summary
import re

from SimpleModule.Quant.Mixed.model.Q_CNN import Q_SimpleCNN

# download tool
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def train_qat(Agent_response, epochs):
    # 解析JSON格式参数
    if Agent_response is None:
        params = {
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_decay": 0.000015,
        "momentum": 0.9,
        "quant_config": {
            "conv1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
            "block": {"x_nbit": 4, "w_nbit": 4, "q_group_size": -1},
            "fc1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
            "fc2": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1}
        }
    }
    else:
        params = json.loads(Agent_response)

    batch_size = params['batch_size']
    EPOCHS = epochs
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    weight_decay = params['weight_decay']
    quant_config = params['quant_config']

    # 基础数据预处理
    print("preprocessing data setting")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 这个给SimpleCNN用
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 下载数据集
    print("download data")
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=1)

    # 显卡设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 训练前设置 - 加载QAT模型，使用混合精度量化
    print("training setting")
    net = Q_SimpleCNN(num_classes=10,
                      conv1_x_nbit=quant_config['conv1']['x_nbit'], conv1_w_nbit=quant_config['conv1']['w_nbit'],
                      conv1_q_group_size=quant_config['conv1']['q_group_size'],
                      block_x_nbit=quant_config['block']['x_nbit'], block_w_nbit=quant_config['block']['w_nbit'],
                      block_q_group_size=quant_config['block']['q_group_size'],
                      fc1_x_nbit=quant_config['fc1']['x_nbit'], fc1_w_nbit=quant_config['fc1']['w_nbit'],
                      fc1_q_group_size=quant_config['fc1']['q_group_size'],
                      fc2_x_nbit=quant_config['fc2']['x_nbit'], fc2_w_nbit=quant_config['fc2']['w_nbit'],
                      fc2_q_group_size=quant_config['fc2']['q_group_size']).to(device)

    # 使用 torchinfo 估算模型参数和运行内存
    input_size = (batch_size, 3, 32, 32)  # CIFAR-10的输入尺寸
    model_summary = summary(net, input_size=input_size)
    summary_str = str(model_summary)
    match = re.search(r"Estimated Total Size \(MB\): (\d+\.\d+)", summary_str)
    if match:
        size_mb = match.group(1)
        m = f"{size_mb}MB"
        print(m)
    else:
        raise ValueError("Memory Data not found in the model summary.")

    # 量化感知训练初始化
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(net, inplace=True)

    # 设置优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=4e-08)

    # 进行量化感知训练
    print("Start QAT training")
    best_acc = 0
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        scheduler.step()

        # 验证量化模型效果
        net.eval()  # 在不转换为全量化模型的情况下验证QAT模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"Validation Accuracy after epoch {epoch + 1}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # 保存量化配置，使用当前时间戳命名文件
            current_time = time.strftime("%Y%m%d_%H%M%S")
            config_filename = f"save/qat_{current_time}.json"
            with open(config_filename, "w") as f:
                json.dump(net.quant_config, f, indent=4)
            print(f"Saved best QAT model config with accuracy: {best_acc:.4f} at {config_filename}")

        net.train()  # 恢复为训练模式

    print("QAT Training Done")
    print("Best validation accuracy: {:.4f}".format(best_acc))

    # 返回最佳准确率和估算的运行内存
    return best_acc, m

if __name__ == '__main__':
    # 示例参数
    params = {
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_decay": 0.000015,
        "momentum": 0.9,
        "quant_config": {
            "conv1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
            "block": {"x_nbit": 4, "w_nbit": 4, "q_group_size": -1},
            "fc1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
            "fc2": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1}
        }
    }

    # 训练QAT模型并获取最佳准确率和运行内存
    best_acc, total_memory = train_qat(params)
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Estimated total memory usage: {total_memory / (1024 ** 2):.2f} MB")
