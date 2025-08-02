#%% - Import thư viện
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#%% Đọc dữ liệu
df = pd.read_csv('D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv')

X = df.drop('PlacementStatus', axis=1)
y = df['PlacementStatus']

#%% Chia dữ liệu thành tập train và test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

#%% - Chạy mô hình với tham số tối ưu cho Decision Tree
print("\nOptimizing Decision Tree with GridSearchCV:")
param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search_dt.fit(X, y)

print("Best parameters (Decision Tree):", grid_search_dt.best_params_)
print("Best F1-Score (Decision Tree):", grid_search_dt.best_score_)

#%% Huấn luyện và đánh giá Decision tree
best_dt_model = grid_search_dt.best_estimator_
best_dt_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred_dt = best_dt_model.predict(X_test)

# Tính toán các chỉ số hiệu suất
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt, zero_division=0)
dt_recall = recall_score(y_test, y_pred_dt, zero_division=0)
dt_f1 = f1_score(y_test, y_pred_dt, zero_division=0)

# Tính confusion matrix
dt_cm = confusion_matrix(y_test, y_pred_dt)

# Hiển thị kết quả
print("\nDecision Tree Performance on Test Set:")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall: {dt_recall:.4f}")
print(f"F1-Score: {dt_f1:.4f}")
print("Confusion Matrix:")
print(dt_cm)

# Vẽ confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix (Decision Tree - Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Vẽ cây quyết định
plt.figure(figsize=(20, 10))
plt.title("Optimized Decision Tree Visualization", fontsize=16, pad=20)
plot_tree(
    best_dt_model,
    feature_names=X.columns,
    class_names=['Not Placed', 'Placed'],
    filled=True,
    rounded=True,
    fontsize=8,
    impurity=True,
    proportion=False,
    precision=3
)
plt.tight_layout()
plt.show()

#%% Tối ưu hóa tham số cho Random Forest (RandomizedSearchCV)
print("\nOptimizing Random Forest with RandomizedSearchCV:")
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
random_search_rf.fit(X, y)

print("Best parameters (Random Forest):", random_search_rf.best_params_)
print("Best F1-Score (Random Forest):", random_search_rf.best_score_)

#%% - Chạy mô hình với tham số tối ưu
# 6. Huấn luyện và đánh giá Random Forest với tham số tối ưu
best_rf_model = random_search_rf.best_estimator_
best_rf_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred_rf = best_rf_model.predict(X_test)

# Tính toán các chỉ số hiệu suất
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)

# Tính confusion matrix
rf_cm = confusion_matrix(y_test, y_pred_rf)

# Hiển thị kết quả
print("\nRandom Forest Performance on Test Set:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")
print("Confusion Matrix:")
print(rf_cm)

# Vẽ confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix (Random Forest - Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()