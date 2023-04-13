from net import Net
import torch
# 加载模型
model = Net()
model.load_state_dict(torch.load('model/trained/model.pth'))

# 将模型设置为评估模式
model.eval()


# input_data 是一个 4 维张量，其形状为 (1, 2, 15, 15)。
# 其中第一个维度表示 batch size，
# 第二个维度表示通道数，第三个和第四个维度表示输入数据的高度和宽度。
# 第二个维度为 2 表示输入数据包含两个通道，分别表示黑子和白子的位置
# 具体数据的含义取决于你的具体情况，但通常来说，它是输入到模型中进行预测的数据。
# 使用模型进行预测
input_data = torch.randn(1, 2, 15, 15)

# output 是模型对 input_data 进行预测后得到的输出结果。
# 具体来说，它是一个张量，其形状和类型取决于你的模型结构和输出层的设置。
# 在这个例子中，由于我们没有提供完整的模型代码，因此无法确定 output 的具体形状和类型。
# 你需要根据你的模型结构和输出层的设置来确定它的具体含义。
# 如果你想查看 output 的具体数值，可以使用 print(output) 或者 output.tolist() 来打印它的值。
# 如果你想将 output 转换为 numpy 数组，可以使用 output.detach().numpy()。
# 如果你想将 output 转换为标量值，可以使用 output.item()。
# 如果你想查看 output 的具体数值，可以使用 print(output) 或者 output.tolist() 来打印它的值。
# 如果你想将 output 转换为 numpy 数组，可以使用 output.detach().numpy()。
# 如果你想将 output 转换为标量值，可以使用 output.item()。

output = model(input_data)