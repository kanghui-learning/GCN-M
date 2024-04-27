import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 10 features to 50
        self.fc2 = nn.Linear(50, 1)   # 50 to 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)

# 随机生成输入和输出数据
inputs = torch.randn(100, 10).to(device)  # 100 samples, 10 features each
targets = torch.randn(100, 1).to(device)  # 100 target values

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):  # loop over the dataset multiple times
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = model(inputs)  # forward pass
    loss = criterion(outputs, targets)  # calculate the loss
    loss.backward()  # backpropagation
    optimizer.step()  # update parameters
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')