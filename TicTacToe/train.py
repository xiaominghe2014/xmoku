import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from net import LSTMModel

# train_data 变量应该是一个输入张量的列表，其中每个张量表示一个单独的游戏棋盘状态。每个张量的形状应该是 (1, 15, 15)，表示单个通道（因为游戏棋盘是灰度的）和一个 15x15 的单元格网格
train_data = np.load('game_data/train_data.npy')

# train_labels 变量应该是一个目标值的列表，其中每个值表示给定 train_data 中相应输入张量的情况下游戏的预期结果。在五子棋的情况下，目标值应该是 1，如果最后一步棋的玩家赢得了比赛，-1，如果另一个玩家赢得了比赛，0，如果比赛以平局结束。
train_labels = np.load('game_data/train_labels.npy')

# 初始化网络和优化器
# 实例化模型
input_size = 3
hidden_size = 128
num_layers = 1
num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 加载模型和优化器的状态
# model.load_state_dict(torch.load('model/model.pth'))
# optimizer.load_state_dict(torch.load('model/optimizer.pth'))

# 更新模型的参数
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data):
        # print(f'data: {data}')
        # print(f'label: {train_labels[i]}')

        data = torch.from_numpy(data)
        data = data.reshape(-1,3,3).to(device)
        label = torch.tensor([train_labels[i]], dtype=torch.long)
        label = label.to(device)
        # 前向传递
        outputs = model(data)
        loss = criterion(outputs, label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印统计信息
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_data), loss.item()))

    # 保存模型和优化器的状态
    torch.save(model.state_dict(), 'model/model.pth')
    torch.save(optimizer.state_dict(), 'model/optimizer.pth')