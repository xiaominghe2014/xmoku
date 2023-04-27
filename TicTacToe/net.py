# 用pytorch 生成一个 PNN概率模型

# 根据需求建立神经网络模型需要考虑多个因素。以下是一些可能有用的建议：

# 1. 确定问题类型：首先，您需要确定您要解决的问题类型。例如，您是在进行分类还是回归？您的数据集是图像、文本还是时间序列数据？

'''
    图像:
        1. 卷积神经网络（Convolutional Neural Network，CNN）：CNN是一种广泛用于图像分类和识别的神经网络模型。它可以自动提取图像中的特征，并将其用于分类或识别任务。

        2. 残差网络（Residual Network，ResNet）：ResNet是一种深度卷积神经网络，它可以有效地解决深度神经网络中的梯度消失问题。它在图像分类和识别任务中表现出色。

        3. 循环神经网络（Recurrent Neural Network，RNN）：RNN是一种广泛用于序列数据处理的神经网络模型。它可以自动提取序列数据中的特征，并将其用于分类或预测任务。
'''

# 2. 选择适当的层数和神经元数：一般来说，您需要根据问题的复杂性选择适当的层数和神经元数。如果您的问题比较简单，您可以使用较少的层数和神经元数。如果您的问题比较复杂，您可能需要使用更多的层数和神经元数。

# 3. 选择适当的激活函数：激活函数对于神经网络的性能非常重要。您需要选择适当的激活函数来确保您的神经网络能够正确地学习。

# 4. 选择适当的损失函数：损失函数用于衡量您的神经网络的性能。您需要选择适当的损失函数来确保您的神经网络能够正确地学习。

# 5. 选择适当的优化器和学习率：优化器和学习率对于神经网络的性能也非常重要。您需要选择适当的优化器和学习率来确保您的神经网络能够正确地学习。

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根据经验，用于下棋游戏双方胜率预测的神经网络模型最好使用卷积神经网络（Convolutional Neural Network，CNN）。CNN 可以有效地处理图像数据，而游戏棋盘可以看作是一个二维图像。因此，使用 CNN 可以更好地提取游戏棋盘的特征，从而更准确地预测双方的胜率。

# 在提取特征方面，可以使用多个卷积层和池化层来逐步提取游戏棋盘的特征。在输出层方面，可以使用一个或多个全连接层来预测双方的胜率。此外，为了防止过拟合，可以在 CNN 中添加一些正则化技术，如 Dropout 和 L2 正则化。

# 以下是一个使用 PyTorch 实现的简单的 CNN 模型，可以作为参考

# 构建模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 卷积华
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)  # 修改隐藏层为三层
        self.fc3 = nn.Linear(64, 3)  # 修改输出层为一个三元组

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 将输入张量 x 的形状变为 (-1, 64 3 3)。
        # 其中 -1 表示该维度的大小由其他维度推断得出，
        # 64 3 3 表示该张量的总元素个数为 64 3 3 个。
        # 这个操作通常用于将卷积层的输出张量展平成一维向量，以便于后续的全连接层处理
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x)) # 将输入 x 传入第一个全连接层 fc1，然后使用 ReLU 激活函数进行非线性变换
        x = F.relu(self.fc2(x))  # 隐藏层为三层
        x = self.fc3(x)  # 输出一个三元组
        return x