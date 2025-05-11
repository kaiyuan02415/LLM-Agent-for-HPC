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
import time
from torchinfo import summary
import re
import os
import json

# download tool
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from LLM_Agent_for_HPC.LLM_Agent.Quant.model.Q_CNN import Q_SimpleCNN
from LLM_Agent_for_HPC.LLM_Agent.Quant.config import config
from LLM_Agent_for_HPC.LLM_Agent.Quant.Prompt import Prompt0

# My openai-key
key = My-Key

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
def train_qat(Agent_response, epochs, Timer):
    # 解析JSON格式参数
    if Agent_response is None:
        params = {
        "learning_rate": 0.001,
        "batch_size": 128,
        "weight_decay": 0.000015,
        "momentum": 0.9,
        "quant_config": {
            "conv1": {"x_nbit": 8, "w_nbit": 8, "q_group_size": -1},
            "blocks": {"x_nbit": 4, "w_nbit": 4, "q_group_size": -1},
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
                      block_x_nbit=quant_config['blocks']['x_nbit'], block_w_nbit=quant_config['blocks']['w_nbit'],
                      block_q_group_size=quant_config['blocks']['q_group_size'],
                      fc1_x_nbit=quant_config['fc1']['x_nbit'], fc1_w_nbit=quant_config['fc1']['w_nbit'],
                      fc1_q_group_size=quant_config['fc1']['q_group_size'],
                      fc2_x_nbit=quant_config['fc2']['x_nbit'], fc2_w_nbit=quant_config['fc2']['w_nbit'],
                      fc2_q_group_size=quant_config['fc2']['q_group_size']).to(device)

    # 使用torchinfo估算模型参数和运行内存
    if Timer == config.get("Max_time", 10):
        input_size = (batch_size, 3, 32, 32)  # CIFAR-10的输入尺寸
        model_summary = summary(net, input_size=input_size)
        summary_str = str(model_summary)
        match = re.search(r"Estimated Total Size \(MB\): (\d+\.\d+)", summary_str)
        if match:
            size_mb = match.group(1)
            m = f"{size_mb}MB"
            print("Estimated Memory : {}".format(m))
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
    return best_acc

def cs_function(args):
    conversation_count = 0
    max_conversations = args.get("Max_time", 10)
    model = args.get("model", "gpt-3.5-turbo")
    print("model type : {}".format(model))

    if conversation_count == 0:
        connection_test(model)

    # 初始化对话记录
    messages = [{"role": "system",
                 "content": "You are an expert assistant specialized in tuning parameters"
                            "for quantized neural networks"
                            "Your goal is to maximize the accuracy of the quantized model"
                            "under the condition of given running memory."}]
    conversation_log = []
    recent_messages = []

    # 调用Openai-api时必须显式进行上下文管理
    # 第一次对话使用Prompt0
    messages.append({"role": "user", "content": Prompt0})
    recent_messages.append({"role": "user", "content": Prompt0})

    while conversation_count < max_conversations:
        print(f"正在进行第 {conversation_count + 1} 轮，还剩 {max_conversations - conversation_count - 1} 轮。")

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
        training_output= train_qat(Agent_response=json.dumps(current_config), epochs=args.get("epochs", 100), Timer=conversation_count)

        # 构建 Prompt_iter 用于下一次对话
        prompt_iter = f"""
        目前的配置为：
        {json.dumps(current_config, indent=4)}
        基于此配置的结果：
        验证正确率：{training_output}
        请提供一组优化后的配置
        """

        # 记录并保留最近三次的对话记录
        if len(recent_messages) > 6:  # 每轮两条消息：用户输入和AI响应
            recent_messages = recent_messages[-6:]

        # 更新 messages 为 Prompt0 + 最近三轮对话
        # 保留 Prompt0 并始终包含在消息记录中
        messages = [
            {"role": "system",
             "content": "You are an expert assistant specialized in tuning parameters "
                        "for quantized neural networks "
                        "Your goal is to maximize the accuracy of the quantized model "
                        "under the condition of given running memory."},
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
