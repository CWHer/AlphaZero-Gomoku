## 说明文档

### Features

- [x] Residual Network
- [x] Monte Carlo Tree Search (with network)
- [ ] Monte Carlo Tree Search (without network)
- [x] Set proper rewards
- [x] Multiprocessing
- [ ] TensorBoard real-time monitoring
- [x] Single player mode
- [x] Two player mode
- [ ] comments
- [ ] How to share model between different processes
- [ ] ...



### 代码架构

大致如下图所示（不完全一致）

Self Play阶段使用CPU inference（~~显存不够~~）

- [ ] 在Linux下可以用GPU inference，但是每个process都会copy一份model

  在Windows下直接报错，只能用CPU inference

![](assets/code-structure.png)

![](assets/alpha_go_zero_cheat_sheet.png)



### 文件说明

| Agent文件夹      |                     |
| ---------------- | ------------------- |
| mcts.py          | MCTS (with network) |
| mcts_utils.py    | PUCT, TreeNode类    |
| network.py       | PolicyValueNet      |
| network_utils.py | 特征选择(encoder)   |

| Env文件夹    |            |
| ------------ | ---------- |
| simulator.py | 模拟五子棋 |

| Train_utils文件夹 |                |
| ----------------- | -------------- |
| game.py           | 自我博弈，对战 |
| replay_buffer.py  | 存储数据       |

| File            | Description          |
| --------------- | -------------------- |
| train.py (main) | 训练pipeline         |
| utils.py        | 可视化及其它辅助函数 |
| config.py       | 超参数设置           |



### 训练Pipeline

1. Self Play生成数据，保存在replay buffer
2. 数据量足够后开始训练model
3. 几轮训练后与best net对局，胜率>55%则更新模型



### 特征选取

参考AlphaZero

