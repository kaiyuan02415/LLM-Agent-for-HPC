import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from models.SimpleNN import SimpleNN

# download tool
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def train():
    # 基础数据预处理
    print("preprocessing data setting")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("download data")
    # 下载数据集
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=1)

    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, num_workers=1)

    # 数据集设置
    print("data setting")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 显卡设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 训练前设置
    print("training setting")
    net = SimpleNN(10).to('cuda')
    cp = os.getcwd()
    print('current location : {}'.format(cp))
    save_path = os.path.join(cp,"save/res.pth")
    print('save_path : {}'.format(save_path))
    # save_path = cp + "save/res.pth"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0006, momentum=0.9, weight_decay=0.0004)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    # 这个暂时可以不告诉Agent
    global_len = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=4e-08)

    # 计算参数量
    # params = list(net.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))

    # 打印模型中的可学习参数数量和不可习参数数量
    # num_learnable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # num_non_learnable_params = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    # print(f"Number of learnable parameters: {num_learnable_params}")
    # print(f"Number of non-learnable parameters: {num_non_learnable_params}")

    # 训练
    print("train Net")
    EPOCHS = 60
    lossList = []
    accList = []
    best_acc = 0
    net.train()
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f'Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)
        lossList.append(avg_loss)

        # 验证
        net.eval()
        correct_res = 0
        total_num = 0
        acc = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total_num += labels.size(0)
                correct_res += (predicted == labels).sum().item()
                acc = correct_res / total_num
        accList.append(acc)
        print("acc : {}".format(acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),save_path)
            print("saved a new pth")

    print('Training Done')
    print("Best acc : {}".format(best_acc))

    # 记录loss和acc
    with open("save/output_loss.txt", "w") as file:
        # 遍历列表中的每一项并写入文件
        for item in lossList:
            file.write(f"{item}\n")
    with open("save/output_acc.txt", "w") as file:
        # 遍历列表中的每一项并写入文件
        for item in accList:
            file.write(f"{item}\n")

    draw_loss(lossList, EPOCHS)

# 画图
def draw_loss(Loss_list, epoch):
    plt.cla()
    x1 = range(1, epoch + 1)
    print(x1)
    y1 = Loss_list
    print(y1)
    plt.title('Loss layout', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train()
