#%% Import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, log_loss
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#%% Đọc dữ liệu
df = pd.read_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv")
print(df.head())
#%% Config
plt.figure(figsize=(10, 6))

#%% Vẽ ma trận tương quan
corr_matrix = df.corr()

# Vẽ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()


#%% RFE + Cross-Validation
# Dữ liệu chỉ chứa các biến được chọn từ heatmap
X_selected = df[["Projects", "AptitudeTestScore", "ExtracurricularActivities", "SSC_Marks", "HSC_Marks"]]
y = df["PlacementStatus"]

# Khởi tạo mô hình hồi quy logistic
model = LogisticRegression(solver="liblinear")

# Dùng StratifiedKFold để đảm bảo phân phối lớp cân bằng
cv = StratifiedKFold(n_splits=5)

# RFE với Cross-Validation
selector = RFECV(model, step=1, cv=cv, scoring="accuracy")
selector.fit(X_selected, y)

# Xem những biến thực sự quan trọng theo RFE
selected_features_rfe = X_selected.columns[selector.support_]

print("Biến được chọn theo RFE + Cross-Validation:", list(selected_features_rfe))
print("Số lượng biến tối ưu:", selector.n_features_)

# Vẽ biểu đồ số lượng biến VS. độ chính xác cross-validation
plt.figure(figsize=(10, 6))
plt.xlabel("Số lượng biến được chọn")
plt.ylabel("Độ chính xác (Cross-Validation Score)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
         selector.cv_results_['mean_test_score'],
         marker='o', linestyle='--', color='b')
plt.title("Số lượng biến VS. Độ chính xác")
plt.grid(True)
plt.show()

#%% VIF
X_selected = df[["AptitudeTestScore", "ExtracurricularActivities", "HSC_Marks"]]

# Thêm một cột hằng số để tính VIF
X_vif = X_selected.copy()
X_vif["Intercept"] = 1  # Cần thiết cho tính toán VIF

# Tính VIF cho từng biến
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Loại bỏ hàng chứa "Intercept" vì ta không quan tâm đến nó
vif_data = vif_data[vif_data["Variable"] != "Intercept"]

# Hiển thị kết quả
print("Hệ số VIF của các biến độc lập:")
print(vif_data)

#%% Chia dữ liệu train-test để xây dựng mô hình
X = df[["AptitudeTestScore", "ExtracurricularActivities", "HSC_Marks"]]
y = df["PlacementStatus"]
# Chia dữ liệu 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# hồi quy logistic
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

# Tính AUC-ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# In kết quả
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {roc_auc:.3f}")
print(f"Log-loss: {logloss:.3f}")
# Vẽ đường ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # Đường chéo tham chiếu
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#%%  Dự đoán mô hình
# Dự đoán trên mẫu ngẫu nhiên
random_sample = X.sample(20, random_state=42)
predictions = logreg.predict(random_sample)
predictions_proba = logreg.predict_proba(random_sample)[:, 1]

for i, (pred, prob) in enumerate(zip(predictions, predictions_proba)):
    status = "Được tuyển" if pred == 1 else "Không được tuyển"
    print(f"Ứng viên {i+1}:")
    print(f" - Xác suất được tuyển: {prob:.3f}")
    print(f" - Dự đoán kết quả: {status}\n")

# Trực quan hóa kết quả dự đoán
plt.figure(figsize=(8, 6))
sns.barplot(x=[f'{i+1}' for i in range(len(predictions_proba))],
            y=predictions_proba,
            palette=["green" if p >= 0.5 else "red" for p in predictions_proba])
plt.axhline(0.5, color='gray', linestyle='--', label='Ngưỡng 0.5')
plt.ylim(0, 1)
plt.ylabel("Xác suất được tuyển")
plt.title("Dự đoán tuyển dụng của các ứng viên")
plt.legend()
plt.show()
# Dự đoán trên tập test
