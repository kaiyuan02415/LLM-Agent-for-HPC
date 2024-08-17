代码建设中，初期

Usage：
直接运行LLM_Agent.EXP.main即可，EXP.config可以修改一些基本参数

注意事项：
1.必须用有余额的key才能进行调用
2.必须进行显式上下文管理，和网页chat4不一样，token的消耗可能更大
3.由于每次测试都会消耗token，不妨在本地先运行训练代码保证逻辑无误，我提供了一个范本：SimpleModule.main
4.可能出现无法调用chat4的问题，应该是key的购买问题，我只用了3.5进行实验
