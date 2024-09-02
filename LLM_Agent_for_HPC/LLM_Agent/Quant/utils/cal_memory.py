import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchinfo
import json
import time

from SimpleModule.Quant.Mixed.model.Q_CNN import Q_SimpleCNN

# download tool
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train_qat():
    batch_size = 128
    epoches = 5
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.000015

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
                      conv1_x_nbit=8, conv1_w_nbit=8, conv1_q_group_size=-1,
                      block_x_nbit=4, block_w_nbit=4, block_q_group_size=-1,
                      fc1_x_nbit=8, fc1_w_nbit=8, fc1_q_group_size=-1,
                      fc2_x_nbit=8, fc2_w_nbit=8, fc2_q_group_size=-1).to(device)

    # 模型计算复杂度分析
    print("Model complexity analysis")
    torchinfo.summary(net, input_size=(batch_size, 3, 32, 32))

    # 量化感知训练初始化
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(net, inplace=True)

    # 设置优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=4e-08)

    # 进行量化感知训练
    print("Start QAT training")
    best_acc = 0
    for epoch in range(epoches):
        net.train()
        running_loss = 0.0

        torch.cuda.empty_cache()  # 清空缓存
        start_mem_train = torch.cuda.memory_reserved(device)  # 使用 memory_reserved
        start_time = time.time()

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

        end_mem_train = torch.cuda.memory_reserved(device)  # 使用 memory_reserved
        training_memory_used = (end_mem_train - start_mem_train) / (1024 ** 2)  # MB
        training_time = time.time() - start_time
        print(f"Training memory used after epoch {epoch + 1}: {training_memory_used:.2f} MB")
        print(f"Training time for epoch {epoch + 1}: {training_time:.2f} seconds")

        # 验证量化模型效果
        net.eval()  # 在不转换为全量化模型的情况下验证QAT模型
        correct = 0
        total = 0

        torch.cuda.empty_cache()  # 清空缓存
        start_mem_eval = torch.cuda.memory_reserved(device)  # 使用 memory_reserved
        eval_start_time = time.time()
        with torch.no_grad():
            for i, data in enumerate(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_mem_eval = torch.cuda.memory_reserved(device)  # 使用 memory_reserved
        evaluation_memory_used = (end_mem_eval - start_mem_eval) / (1024 ** 2)  # MB
        evaluation_time = time.time() - eval_start_time
        print(f"Validation memory used after epoch {epoch + 1}: {evaluation_memory_used:.2f} MB")
        print(f"Validation time for epoch {epoch + 1}: {evaluation_time:.2f} seconds")

        acc = correct / total
        print(f"Validation Accuracy after epoch {epoch + 1}: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            # 保存量化配置
            with open("save/qat_best_config.json", "w") as f:
                json.dump(net.quant_config, f, indent=4)
            print("Saved best QAT model config with accuracy: {:.4f}".format(best_acc))

        net.train()  # 恢复为训练模式

    print("QAT Training Done")
    print("Best validation accuracy: {:.4f}".format(best_acc))

if __name__ == '__main__':
    train_qat()

