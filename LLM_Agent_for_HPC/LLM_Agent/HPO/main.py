import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
import openai
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import os
import json
import re

# download tool
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from LLM_Agent_for_HPC.LLM_Agent.HPO.models.SimpleNN import SimpleNN
from LLM_Agent_for_HPC.LLM_Agent.HPO.models.SimpleNN import SimpleCNN
from LLM_Agent_for_HPC.LLM_Agent.HPO.config import config

# My openai-key
key = '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'

# 初始化的Prompt，参考sklearn格式
Prompt0 = """
给您提供一个神经网络，结构全程保持不变：
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
您正在帮助调整SimpleNN模型的超参数，训练的数据集是CIFAR-10。训练是使用PyTorch进行的。以下是我们的超参数搜索空间：
- `learning_rate`：优化器的学习率。类型：UniformFloat，范围：[1e-05, 1.0]，默认值：0.01，使用对数刻度。
- `batch_size`：每批输入数据的样本数量。类型：UniformInteger，范围：[16, 256]，默认值：128，使用对数刻度。
- `weight_decay`：L2正则化系数。类型：UniformFloat，范围：[1e-08, 0.1]，默认值：0.0001，使用对数刻度。
- `momentum`：用于SGD优化器的动量。类型：UniformFloat，范围：[0.5, 0.99]，默认值：0.9。
- `num_epochs`：训练模型的轮数。类型：UniformInteger，范围：[20, 180]，默认值：60。
您将会在每次尝试后得到准确率。目标是在给定的预算内找到能够最小化错误率的配置。如果损失没有变化，请探索搜索空间的不同部分。
你每次只提供一组配置，当我将训练结果提供给你时，你再返回一组优化后的配置。每次试验的配置和数据都要考虑在内，才能更加全面地分析优化配置
请以JSON格式提供配置。例如：{"learning_rate": x, 
                        "batch_size": y, 
                        "weight_decay": z, 
                        "momentum": w, 
                        "num_epochs": v}
"""

# 参考格式
Prompt_iter = """
目前的配置为：
    "learning_rate": 0.0032,
    "batch_size": 64,
    "weight_decay": 0.00001,
    "momentum": 0.95,
    "num_epochs": 50
}
基于此配置的结果：验证正确率： xxx
请优化并提供一组优化后的配置
"""
# 测试连接
def connection_test(model="gpt-3.5-turbo"):
    try:
        # 设置你的API密钥
        openai.api_key = key

        # 调用ChatGPT
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": "Test input to check connection."}
            ]
        )
        # 检查响应内容是否存在
        if 'choices' in response and len(response['choices']) > 0:
            message = response['choices'][0]['message']['content']
            print("ChatGPT Response:", message)
            return message
        else:
            raise RuntimeError("Response error")

    except openai.error.AuthenticationError:
        print("Invalid key")
    except openai.error.OpenAIError as e:
        print(f"API call failure：{e}")
    except ValueError as e:
        print(f"Invalid input：{e}")
    except Exception as e:
        print(f"Unknown error：{e}")

    print("Connection Established")


# 训练函数，传参包括配置信息
def train(Agent_response, timer, class_number=10):
    # 解析重要信息JSON
    if Agent_response is None:
        important_info = {
            "learning_rate": 0.005,
            "batch_size": 128,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "num_epochs": 50
        }
    else:
        important_info = json.loads(Agent_response)

    # 提取参数
    learning_rate = important_info.get("learning_rate", 0.005)
    batch_size = important_info.get("batch_size", 128)
    weight_decay = important_info.get("weight_decay", 0.0001)
    momentum = important_info.get("momentum", 0.9)
    num_epochs = important_info.get("num_epochs", 50)

    # 打印本次内容
    print("iter number : {}".format(timer))
    print("config this iter : \nlearning_rate : {0}\nbatch_size : {1}\nweight_decay : {2}\nmomentum : {3}\nnum_epochs "
          ": {4}".format(learning_rate, batch_size, weight_decay, momentum, num_epochs))

    # 基础数据预处理
    print("local : preprocessing data setting")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 后面可以写到内存里面
    print("local : Get data")
    # 下载数据集
    train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=1)

    # 数据集设置
    print("local : data setting")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 显卡设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("local : using {} device.".format(device))

    # 训练前设置
    print("local : training setting")
    net = SimpleNN(10).to(device)
    cp = os.getcwd()
    print('local : current location : {}'.format(cp))
    save_path = os.path.join(cp,"save/res.pth")
    print('local : save_path : {}'.format(save_path))
    # save_path = cp + "save/res.pth"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=4e-08)

    # 训练
    print("local : train Net")
    EPOCHS = num_epochs
    lossList = []
    accList = []
    best_acc = 0
    net.train()
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0 and i > 0:
                print(f'local : Loss [{epoch + 1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        avg_loss = sum(losses) / len(losses)
        scheduler.step()
        # scheduler.step(avg_loss)
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
        print("local : acc : {}".format(acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(),save_path)
            print("local : saved a new pth")
    print('local : Training Done')

    # 记录loss和acc
    with open("save/output_loss.txt", "w") as file:
        # 遍历列表中的每一项并写入文件
        for item in lossList:
            file.write(f"{item}\n")
    with open("save/output_acc.txt", "w") as file:
        # 遍历列表中的每一项并写入文件
        for item in accList:
            file.write(f"{item}\n")

    # 返回最佳准确率和Prompt所需的list
    return best_acc, lossList, accList
def cs_function(args):
    conversation_count = 0
    max_conversations = args.get("Max_time", 10)
    current_config = None
    model = args.get("model", "gpt-3.5-turbo")
    print("model type : {}".format(model))

    if conversation_count == 0:
        connection_test(model)

    # 初始化对话记录
    messages = [{"role": "system", "content": "You are an expert assistant specialized in optimizing hyperparameters for neural networks. "
                                              "Your goal is to help improve the performance of neural networks by providing optimized hyperparameter configurations."}]
    conversation_log = []
    recent_messages = []

    # 调用Openai-api时必须显式进行上下文管理
    # 第一次对话使用Prompt0
    messages.append({"role": "user", "content": Prompt0})
    recent_messages.append({"role": "user", "content": Prompt0})

    while conversation_count < max_conversations:
        print(f"正在进行第 {conversation_count + 1} 轮，还剩 {max_conversations - conversation_count-1} 轮。")

        # 调用ChatGPT API获取配置
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages  # 发送消息历史
            )
        except Exception as e:
            print(f"调用OpenAI API失败: {str(e)}")
            break

        # 提取GPT的回复内容，并解析为新的配置
        server_response = response['choices'][0]['message']['content']
        print(f"服务器提供的配置:\n{server_response}\n")

        try:
            json_string = re.search(r'\{.*\}', server_response, re.DOTALL).group(0)
            # 假设GPT返回的是一个有效的JSON配置
            current_config = json.loads(json_string)
        except json.JSONDecodeError:
            print("无法解析服务器的响应为JSON格式，请检查响应内容。")
            break

        # 记录对话
        conversation_log.append({
            "conversation_round": conversation_count + 1,
            "user_message": messages[-1]['content'],
            "server_response": server_response
        })

        # 将GPT的响应添加到消息记录中
        messages.append({"role": "assistant", "content": server_response})
        recent_messages.append({"role": "assistant", "content": server_response})

        # 使用配置进行训练，并获取验证准确率
        training_output, _, _ = train(Agent_response=json.dumps(current_config), timer=conversation_count+1,
                                      class_number=args["class_number"])

        # 构建 Prompt_iter 用于下一次对话
        prompt_iter = f"""
        目前的配置为：
        {json.dumps(current_config, indent=4)}
        基于此配置的结果：
        验证正确率：{training_output}
        请优化并提供一组优化后的配置
        """

        # 记录并保留最近三次的对话记录
        if len(recent_messages) > 6:  # 每轮两条消息：用户输入和AI响应
            recent_messages = recent_messages[-6:]

        # 更新 messages 为 Prompt0 + 最近三轮对话
        # 保留 Prompt0 并始终包含在消息记录中
        messages = [
            {"role": "system",
             "content": "You are an expert assistant specialized in optimizing hyperparameters for neural networks. "
                        "Your goal is to help improve the performance of neural networks by providing optimized hyperparameter configurations."},
            {"role": "user", "content": Prompt0}  # 始终保留 Prompt0
        ]
        messages.extend(recent_messages)  # 添加最近三轮的对话记录
        messages.append({"role": "user", "content": prompt_iter})  # 添加最新的用户输入
        recent_messages.append({"role": "user", "content": prompt_iter})  # 记录最近的对话

        # 保存每轮的上传 Prompt
        if not os.path.exists('save/Prompt'):
            os.makedirs('save/Prompt')
        with open(f'save/Prompt/round_{conversation_count + 1}_prompt.json', 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

        # 增加对话计数
        conversation_count += 1

    # 保存完整的对话记录到JSON文件
    with open('save/conversation_log.json', 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=4)

    print("对话结束，对话记录已保存到 conversation_log.json 文件中。")

if __name__ == '__main__':
    args = config
    cs_function(args)

