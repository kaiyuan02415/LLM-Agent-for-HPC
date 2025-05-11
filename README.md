这是-用于量化模型微调的Agent-的简单实现

Usage：</br>
（0）运行SimpleModule.main进行本地训练测试</br>
（1）运行LLM_Agent.HPO.main进行基于LLM Agent的超参数调优，EXP.config可以修改一些基本参数</br>
（2）运行LLM_Agent.Quant.main进行基于LLM Agent的量化，支持基本的QAT混合精度量化

注意事项：
1.必须用有余额的key才能进行调用
2.必须进行显式上下文管理，和网页chat4不一样，token的消耗可能更大
3.由于每次测试都会消耗token，不妨在本地先运行训练代码保证逻辑无误，我提供了一个范本：SimpleModule.main

更新：</br>
更新了GPT-4的调用方式，现在支持few-shot的GPT-4的对话方式

解决访问GPT网页出现502 Bad Gateway的问题
1.管理员身份运行cmd
2.输入ipconfig /flushdns
3.输入netsh winsock reset
4.重启电脑即可


