import os
import zipfile
import random
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog

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

# 提取HOG特征
def extract_hog_features(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
        hog_features.append(features)
    return np.array(hog_features)

# 提取HOG特征
train_hog_features = extract_hog_features(train_images)

# 将数据集划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.2, random_state=None)

# 训练SVM模型
svm_model = svm.SVC(kernel='linear', C=1) #原来 C=1
svm_model.fit(X_train, y_train)

# 在验证集上进行预测
val_predictions = svm_model.predict(X_val)

# 计算验证集准确率
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')

# 加载测试集
test_images, test_labels = load_dataset('rps-test-set')

# 提取HOG特征
test_hog_features = extract_hog_features(test_images)

# 在测试集上进行预测
test_predictions = svm_model.predict(test_hog_features)

# 计算测试集准确率
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f'Test Accuracy: {test_accuracy}')


total_images = len(train_hog_features)
validation_set_size = int(0.2 * total_images)
training_set_size = total_images - validation_set_size
test_set_size = len(test_images)


print(f"Training set size: {training_set_size} images")
print(f"Validation set size: {validation_set_size} images")
print(f"Number of images in the test set: {test_set_size}")
