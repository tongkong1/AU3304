import os
import zipfile
import random
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 解压数据集
with zipfile.ZipFile('rps.zip', 'r') as zip_ref:
    zip_ref.extractall('rps')

with zipfile.ZipFile('rps-test-set.zip', 'r') as zip_ref:
    zip_ref.extractall('rps-test-set')

# 准备数据集
def load_dataset(directory):
    images = []
    labels = []
    classes = {'rock': 0, 'paper': 1, 'scissors': 2}

    for class_name, class_label in classes.items():
        class_path = os.path.join(directory, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))  # 调整图像大小
            images.append(img)
            labels.append(class_label)

    return np.array(images), np.array(labels)

# 加载训练集
train_images, train_labels = load_dataset('rps')

# 将数据集划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=None)

# 将数据转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
y_train_tensor = torch.from_numpy(y_train).long()

X_val_tensor = torch.from_numpy(X_val).unsqueeze(1).float()
y_val_tensor = torch.from_numpy(y_val).long()

# 构建CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)  # Dropout before the pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(0.1)  # Dropout before the pooling layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 将数据放入DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 在验证集上进行预测
with torch.no_grad():
    outputs_val = model(X_val_tensor)
    _, predicted_val = torch.max(outputs_val, 1)

# 计算验证集准确率
val_accuracy = accuracy_score(y_val, predicted_val.numpy())
print(f'Validation Accuracy: {val_accuracy}')

# 加载测试集
test_images, test_labels = load_dataset('rps-test-set')

# 将数据转换为PyTorch张量
test_images_tensor = torch.from_numpy(test_images).unsqueeze(1).float()
test_labels_tensor = torch.from_numpy(test_labels).long()

# 在测试集上进行预测
with torch.no_grad():
    outputs_test = model(test_images_tensor)
    _, predicted_test = torch.max(outputs_test, 1)

# 计算测试集准确率
test_accuracy = accuracy_score(test_labels, predicted_test.numpy())
print(f'Final Test Accuracy: {test_accuracy}')

total_images = len(train_images)
validation_set_size = int(0.2 * total_images)
training_set_size = total_images - validation_set_size
test_set_size = len(test_images)

print(f"Training set size: {training_set_size} images")
print(f"Validation set size: {validation_set_size} images")
print(f"Number of images in the test set: {test_set_size}")
