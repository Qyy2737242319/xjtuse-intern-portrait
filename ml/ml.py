import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 生成一个简化的用户行为数据集
np.random.seed(42)
data = pd.DataFrame({
    'last_purchase_time': np.random.randint(1, 30, 1000),
    'last_login_time': np.random.randint(1, 30, 1000),
    'purchase_amount': np.random.normal(100, 20, 1000),
    'purchase_cycle': np.random.randint(1, 30, 1000),
    'ip_address': np.random.randint(1, 100, 1000),
    'purchase_product': np.random.randint(1, 5, 1000),
    'is_fraud': np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
})

# 添加一些异常值
data.loc[data['is_fraud'] == 1, 'purchase_amount'] += np.random.normal(200, 50, sum(data['is_fraud']))

# 分割数据集为特征和标签
X = data[['last_purchase_time', 'last_login_time', 'purchase_amount', 'purchase_cycle', 'ip_address', 'purchase_product']].values
y = data['is_fraud'].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 2. 模型构建
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 创建模型实例
input_size = X_train.shape[1]
model = FraudDetectionModel(input_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 模型训练
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

# 4. 模型评估
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = (y_pred_tensor.numpy() > 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)

print("Accuracy: {:.2f}%".format(accuracy * 100))
