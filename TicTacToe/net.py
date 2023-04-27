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

# 构建模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device).float()
        x = x.to(torch.float32)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out