# https://www.cnblogs.com/bohengwebb/p/18718114
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_samples = 100
data = torch.randn(n_samples, 2)
# 这行代码的目的是为每个生成的二维数据点生成一个标签，用于表示该点是否位于单位圆内。点在圆内为1，圆外为0。
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)

plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

class SimpleNN(nn.Module): 
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(2, 4)
    self.fc2 = nn.Linear(4, 4)
    self.fc3 = nn.Linear(4, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.sigmoid(self.fc3(x))
    return x

model = SimpleNN()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 50000
for epoch in range(epochs):
  outputs = model(data)
  loss = criterion(outputs, labels)
  # 反向传播
  optimizer.zero_grad()   # 梯度清零 
  # 在每次进行反向传播之前，需要将之前计算的梯度清零，以避免梯度累加。这是因为在 PyTorch 中，梯度是累加的，即每次调用 backward() 方法时，梯度会被累加到之前的梯度上。
  # 因此，在每次迭代开始时，需要将梯度清零，以确保每次计算的梯度是当前批次数据的梯度。
  loss.backward()         # 反向传播计算梯度
  # 用于执行反向传播，计算损失函数关于模型参数的梯度。
  # backward() 方法会自动计算损失函数关于模型参数的梯度，并将这些梯度存储在模型参数的 .grad 属性中。
  optimizer.step()        # 更新参数 用于根据计算得到的梯度更新模型参数。
  # step() 方法会根据优化器的更新规则（例如学习率、动量等），根据计算得到的梯度更新模型的参数。
  # 反向传播，可以理解为“向后算”。从输出层开始，一步步根据损失函数来修正模型、优化模型，算出优化后的模型内参数（也就是前文提到的矩阵Wfcx与向量bfcx)

  if (epoch + 1) % 1000 == 0:
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 保存整个模型
torch.save(model, 'model1.pth')

def plot_decision_boundary(model, data):
  x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
  y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
  xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1), indexing='ij')
  grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
  predictions = model(grid).detach().numpy().reshape(xx.shape)
  plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)
  plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm', edgecolors='k')
  plt.title("Decision Boundary")
  #plt.show()
  plt.savefig("result.png")

plot_decision_boundary(model, data)