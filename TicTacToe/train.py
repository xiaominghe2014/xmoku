import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from net import CNNModel

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
model = CNNModel().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 加载模型和优化器的状态
# model.load_state_dict(torch.load('model/model.pth'))
# optimizer.load_state_dict(torch.load('model/optimizer.pth'))

def train(net, train_loader, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = torch.from_numpy(data).float()
        labels = torch.tensor([train_labels[i]], dtype=torch.long)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        inputs = torch.unsqueeze(inputs, 0)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# 更新模型的参数
num_epochs = 30
for epoch in range(num_epochs):
    train_loss = train(model, train_data, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # 保存模型和优化器的状态
    torch.save(model.state_dict(), 'model/model.pth')
    torch.save(optimizer.state_dict(), 'model/optimizer.pth')