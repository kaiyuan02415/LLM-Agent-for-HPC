# 简介
这是一个用于量化模型微调的 Agent 的简单实现。

---

## 用法

1. **本地训练测试**  
   运行 `SimpleModule.main` 进行本地训练测试。

2. **超参数调优**  
   运行 `LLM_Agent.HPO.main`，基于 LLM Agent 进行超参数调优。可通过 `EXP.config` 修改基本配置参数。

3. **模型量化**  
   运行 `LLM_Agent.Quant.main`，基于 LLM Agent 进行量化，支持 QAT（混合精度量化）。

---

## 注意事项

- **Key 要有余额**：必须使用余额充足的 API Key 才能正常调用。  
- **显式上下文管理**：与网页版 ChatGPT 不同，请务必手动管理对话上下文，避免意外断流，且 token 消耗可能更大。  
- **本地验证**：每次测试都会消耗 token，建议先在本地运行 `SimpleModule.main`，确保逻辑正确后再调用 Agent。

---

## 更新日志

1. **GPT-4 调用方式更新**  
   - 现已支持 Few-shot 对话模式。
   - 现已支持 Chunk 流式对话模式。

2. **502 Bad Gateway 问题修复**  
   如遇到访问 GPT 出现 “502 Bad Gateway” 错误，请按以下步骤操作：  
   ```bash
   (1)
   以管理员身份打开命令提示符:
   ipconfig /flushdns
   netsh winsock reset
   (2) 重启电脑



# **English Version**

# Overview
A lightweight Agent implementation for quantization-aware model fine-tuning.

---

## Usage

1. **Local Training & Testing**  
   Run `SimpleModule.main` to perform local training and validation.

2. **Hyperparameter Optimization**  
   Run `LLM_Agent.HPO.main` for LLM Agent–based hyperparameter tuning. Adjust basic settings via `EXP.config`.

3. **Model Quantization**  
   Run `LLM_Agent.Quant.main` to perform quantization with the LLM Agent, including QAT (quantization-aware training).

---

## Notes

- **API Key Balance**  
  Ensure your API key has sufficient balance for calls.

- **Explicit Context Management**  
  Unlike the web ChatGPT, you must manually manage the conversation context. Token consumption may be higher.

- **Local Verification**  
  Each test consumes tokens. It’s recommended to verify the workflow locally by running `SimpleModule.main` before using the Agent.

---

## Changelog

1. **Updated GPT-4 Invocation**  
   - Added support for few-shot conversation mode.
   - Added support for stream conversation mode

2. **Fixed 502 Bad Gateway Error**  
   If you encounter a “502 Bad Gateway” when accessing GPT, execute the following in an administrator command prompt:
   ```bash
   ipconfig /flushdns
   netsh winsock reset
   # Then reboot your machine
