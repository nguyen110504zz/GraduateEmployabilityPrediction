#%% Import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

#%% Đọc dữ liệu
df = pd.read_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv")

# Tách dữ liệu thành X và y
X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

#%% Chia dữ liệu theo tỷ lệ 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình mạng nơ-ron
def create_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(X.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình với tập train-test (80-20)
model = create_model()
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=0)

# Dự đoán
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Đánh giá mô hình với 80-20
print("=== Kết quả với tập train-test (80-20) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Vẽ biểu đồ Loss và Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.show()

# Vẽ Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Vẽ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # Đường chéo tham chiếu
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%% Huấn luyện mô hình với K-Fold Cross Validation (K=5)
# Hàm tạo mô hình với Input()
def create_model():
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
all_fpr, all_tpr, all_roc_auc = [], [], []
all_cm = np.zeros((2, 2))  # Tổng Confusion Matrix
history_list = []  # Lưu lại history của từng fold

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    model_kfold = create_model()
    history = model_kfold.fit(X_train_fold, y_train_fold, epochs=50, batch_size=10, verbose=0, validation_data=(X_val_fold, y_val_fold))
    history_list.append(history.history)  # Lưu lại history của fold này

    y_val_pred_prob = model_kfold.predict(X_val_fold)
    y_val_pred = (y_val_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_val_fold, y_val_pred)
    cv_scores.append(acc)

    # Lưu Confusion Matrix
    cm = confusion_matrix(y_val_fold, y_val_pred)
    all_cm += cm  # Cộng dồn để tính trung bình

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_pred_prob)
    roc_auc = auc(fpr, tpr)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

# Tính loss và accuracy trung bình của các folds
avg_loss = np.mean([h['loss'] for h in history_list], axis=0)
avg_val_loss = np.mean([h['val_loss'] for h in history_list], axis=0)
avg_acc = np.mean([h['accuracy'] for h in history_list], axis=0)
avg_val_acc = np.mean([h['val_accuracy'] for h in history_list], axis=0)

# Vẽ biểu đồ Loss & Accuracy trung bình
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(avg_loss, label='Train Loss')
plt.plot(avg_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs (K-Fold)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(avg_acc, label='Train Accuracy')
plt.plot(avg_val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs (K-Fold)')
plt.legend()

plt.show()

# Vẽ Confusion Matrix trung bình
plt.figure(figsize=(6, 5))
sns.heatmap(all_cm / 5, annot=True, fmt=".1f", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Average Confusion Matrix (K-Fold)")
plt.show()

# Vẽ ROC Curve trung bình
plt.figure(figsize=(8, 6))
for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], alpha=0.3, label=f"Fold {i+1} (AUC = {all_roc_auc[i]:.3f})")

plt.plot([0, 1], [0, 1], 'k--')  # Đường chéo tham chiếu
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (K-Fold)')
plt.legend()
plt.show()