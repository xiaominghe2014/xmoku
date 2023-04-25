import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from net import Net
# 定义训练数据和标签

# train_data 变量应该是一个输入张量的列表，其中每个张量表示一个单独的游戏棋盘状态。每个张量的形状应该是 (1, 15, 15)，表示单个通道（因为游戏棋盘是灰度的）和一个 15x15 的单元格网格
train_data = np.load('game_data/train_data.npy')

# train_labels 变量应该是一个目标值的列表，其中每个值表示给定 train_data 中相应输入张量的情况下游戏的预期结果。在五子棋的情况下，目标值应该是 1，如果最后一步棋的玩家赢得了比赛，-1，如果另一个玩家赢得了比赛，0，如果比赛以平局结束。
train_labels = np.load('game_data/train_labels.npy')

# 初始化网络和优化器
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 定义损失函数
criterion = nn.MSELoss()

# 加载模型和优化器的状态
net.load_state_dict(torch.load('model/trained/model.pth'))
optimizer.load_state_dict(torch.load('model/trained/optimizer.pth'))

# 更新模型的参数
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data):
        # 将参数梯度归零
        optimizer.zero_grad()

        # 前向传递
        outputs = net(data)
        loss = criterion(outputs, train_labels[i])

        # 反向传递和优化
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # 保存模型和优化器的状态
    torch.save(net.state_dict(), 'model/trained/model.pth')
    torch.save(optimizer.state_dict(), 'model/trained/optimizer.pth')