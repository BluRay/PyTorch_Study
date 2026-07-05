# https://www.runoob.com/pytorch/pytorch-linear-regression.html
# PyTorch 线性回归
# 线性回归是最基本的机器学习算法之一，用于预测一个连续值。它是一种简单且常见的回归分析方法，目的是通过拟合一个线性函数来预测输出。
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 准备数据
torch.manual_seed(42)
X = torch.randn(100, 3)  # 100 个样本，3个特征
true_w = torch.tensor([2.0, 3.0, 4.0])
true_b = 5.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

# 2. 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. 定义损失函数和优化器
# 损失函数（均方误差）
criterion = nn.MSELoss()
# 优化器（使用 SGD 或 Adam） SGD:使用随机梯度下降法更新参数，学习率控制每步更新的幅度
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 学习率设置为0.01
# 也可以使用 Adam 优化器     Adam:自适应学习率优化器，通常收敛更快
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    model.train()

    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
print(f'\n训练后的权重: {model.linear.weight.data.numpy()}')
print(f'训练后的偏置: {model.linear.bias.data.numpy()}')
print(f'真实权重: {true_w.numpy()}')
print(f'真实偏置: {true_b}')