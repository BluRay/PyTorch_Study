# https://www.cnblogs.com/bohengwebb/p/18718114
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_samples = 300
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

model = torch.load("model1.pth", weights_only=False)

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