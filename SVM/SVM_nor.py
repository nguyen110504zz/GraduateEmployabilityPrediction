#%% Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

#%% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv"
df = pd.read_csv(file_path)

#%% Tách biến độc lập và biến mục tiêu
X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

#%% Chia dữ liệu thành tập train (80%) và test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% Giảm số chiều xuống 2D bằng PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#%% Khởi tạo và huấn luyện mô hình SVM
svm_model = SVC(kernel="rbf", C=10, gamma=1, random_state=42)
svm_model.fit(X_train_pca, y_train)

#%% Dự đoán trên tập test
y_pred = svm_model.predict(X_test_pca)

#%% Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#%% In kết quả
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

#%% Vẽ ma trận nhầm lẫn (Confusion Matrix)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Placed", "Placed"], yticklabels=["Not Placed", "Placed"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - SVM Model")
plt.show()

#%% Vẽ đường biên quyết định trên 2D PCA
plt.figure(figsize=(10, 6))

# Tạo lưới điểm để vẽ đường biên
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Dự đoán trên lưới điểm meshgrid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ đường biên quyết định
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Vẽ dữ liệu thực tế
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='bwr', edgecolor='k', alpha=0.8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Decision Boundary of SVM in 2D PCA Space")
plt.legend(handles=scatter.legend_elements()[0], labels=["Not Placed", "Placed"])
plt.show()
