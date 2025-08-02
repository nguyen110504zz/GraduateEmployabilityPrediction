#%% Import thư viện
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

#%% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/aftermining_placementdata.csv"
df = pd.read_csv(file_path)

#%% Loại bỏ dữ liệu trùng lặp
df = df.drop_duplicates()
df.info()

#%% Chọn các biến độc lập theo phân tích
selected_features = ["CGPA", "Projects", "AptitudeTestScore",
                     "SoftSkillsRating", "ExtracurricularActivities",
                     "SSC_Marks", "HSC_Marks"]

X = df[selected_features]  # Chỉ giữ lại các biến đã chọn
y = df["PlacementStatus"]  # Biến mục tiêu

#%% Vẽ Boxplot trước khi xử lý Outliers
plt.figure(figsize=(15, 8))
for i, col in enumerate(selected_features):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f"Before Outlier Removal: {col}")
plt.tight_layout()
plt.show()

#%% Chuẩn hóa dữ liệu bằng Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[selected_features]), columns=selected_features)

# Ghép với biến mục tiêu
df_scaled = pd.concat([X_scaled, df["PlacementStatus"].reset_index(drop=True)], axis=1)

#%% Cân bằng dữ liệu bằng SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

#%% Chuyển dữ liệu về DataFrame
df_final = pd.DataFrame(X_resampled, columns=selected_features)
df_final["PlacementStatus"] = y_resampled.astype(int)

#%% Shuffle dữ liệu
df_final = shuffle(df_final, random_state=42).reset_index(drop=True)

#%% Vẽ lại phân phối dữ liệu
plt.figure(figsize=(6, 4))
sns.countplot(x=df_final["PlacementStatus"], palette=["red", "green"])

# Hiển thị số lượng từng nhóm
for i, value in enumerate(df_final["PlacementStatus"].value_counts().values):
    plt.text(i, value + 50, str(value), ha="center", fontsize=12)

plt.title("Tỷ lệ sinh viên được tuyển và không được tuyển (Sau SMOTE)")
plt.xlabel("Placement Status")
plt.ylabel("Số lượng")
plt.xticks(ticks=[0, 1], labels=["Không được tuyển", "Được tuyển"])
plt.show()

#%% Xuất dữ liệu đã xử lý
df_final.to_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv", index=False)

#%% Kiểm tra thông tin dữ liệu
df_final.info()