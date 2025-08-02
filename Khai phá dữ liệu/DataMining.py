#%% Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Đọc dữ liệu
df = pd.read_csv('D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/placementdata.csv')

#%% Trích xuất thông tin dữ liệu
print('Thông tin dữ liệu:')
df.info()

#%% In ra giá trị của các cột có kiểu dữ liệu object
unique_values = {col: df[col].unique() for col in ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']}
print(unique_values)

#%% Loại bỏ cột không cần thiết
df = df.drop(columns=['StudentID'], errors='ignore')

#%% Mô tả dữ liệu
print('\nMô tả dữ liệu:')
pd.set_option('display.max_columns', None)
print(df.describe(include='all'))

#%% Chuyển đổi biến phân loại thành dạng số
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'Yes': 1, 'No': 0})
df['PlacementTraining'] = df['PlacementTraining'].map({'Yes': 1, 'No': 0})
df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})

#%% Thống kê mô tả của CGPA
print("\nMô tả thống kê của CGPA:")
print(df['CGPA'].describe())

#%%So sánh số liệu giữa 2 nhóm
# Chia thành 2 nhóm
placed = df[df['PlacementStatus'] == 1]
not_placed = df[df['PlacementStatus'] == 0]

# Tạo bảng thống kê mô tả
summary_table = pd.DataFrame({
    "Overall": df['CGPA'].describe(),
    "Placed": placed['CGPA'].describe(),
    "Not Placed": not_placed['CGPA'].describe()
})

# Hiển thị bảng
print("\nBảng so sánh các chỉ số CGPA giữa hai nhóm tuyển dụng:")
print(summary_table)

#%% Cấu hình biểu đồ
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

sns.boxplot(x=df["CGPA"], ax=axes[0, 1])
axes[0, 1].set_title("Boxplot CGPA")

sns.histplot(df["CGPA"], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Phân phối CGPA")
sns.histplot(not_placed["CGPA"], bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Phân phối CGPA của nhóm không được tuyển")

sns.histplot(placed["CGPA"], bins=30, kde=True, ax=axes[1, 2])
axes[1, 2].set_title("Phân phối CGPA của nhóm được tuyển")
plt.tight_layout()
plt.show()

#%% Tạo bảng thống kê số lượng và tỷ lệ sinh viên theo số lượng thực tập và trạng thái tuyển dụng
internship_placement_counts = df.groupby(["Internships", "PlacementStatus"]).size().unstack(fill_value=0)

# Tính tổng số sinh viên theo từng mức thực tập
internship_placement_counts["Total"] = internship_placement_counts.sum(axis=1)

# Tính tỷ lệ sinh viên được tuyển và không được tuyển
internship_placement_counts["Not Placed (%)"] = (internship_placement_counts[0] / internship_placement_counts["Total"]) * 100
internship_placement_counts["Placed (%)"] = (internship_placement_counts[1] / internship_placement_counts["Total"]) * 100

# Xuất bảng thống kê
print("\nBảng thống kê số lượng và tỷ lệ sinh viên theo số lượng thực tập:")
print(internship_placement_counts)

#%% Vẽ biểu đồ cột số lượng sinh viên được tuyển và không được tuyển theo số lượng thực tập
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Biểu đồ 1: Tổng số sinh viên theo số lượng thực tập
sns.barplot(x=internship_placement_counts.index, y=internship_placement_counts["Total"], color="red", ax=axes[0])
axes[0].set_title("Tổng số sinh viên theo số lượng thực tập")
axes[0].set_xlabel("Số lượng thực tập")
axes[0].set_ylabel("Số lượng sinh viên")

# Biểu đồ 2: Phân bố sinh viên được tuyển và không được tuyển
internship_placement_counts[[0, 1]].plot(kind="bar", stacked=True, color=["red", "green"], ax=axes[1])
axes[1].set_title("Số lượng sinh viên được tuyển và không được tuyển")
axes[1].set_xlabel("Số lượng thực tập")
axes[1].set_ylabel("Số lượng sinh viên")
axes[1].legend(["Không được tuyển", "Được tuyển"])

# Biểu đồ 3: Tỷ lệ phần trăm sinh viên được tuyển và không được tuyển
internship_placement_counts[["Not Placed (%)", "Placed (%)"]].plot(kind="bar", stacked=True, color=["red", "green"], ax=axes[2])
axes[2].set_title("Tỷ lệ phần trăm sinh viên được tuyển và không được tuyển")
axes[2].set_xlabel("Số lượng thực tập")
axes[2].set_ylabel("Tỷ lệ (%)")
axes[2].legend(["Không được tuyển", "Được tuyển"])

plt.tight_layout()
plt.show()

# Hiển thị bảng
print("\nBảng so sánh các chỉ số CGPA giữa hai nhóm tuyển dụng:")
print(summary_table)

#%%  Vẽ biểu đồ cột số lượng sinh viên được tuyển và không được tuyển theo số lượng dự án
project_placement_counts = df.groupby(["Projects", "PlacementStatus"]).size().unstack(fill_value=0)

# Tính tổng số sinh viên theo số lượng dự án
project_placement_counts["Total"] = project_placement_counts.sum(axis=1)

# Tính tỷ lệ sinh viên được tuyển và không được tuyển
project_placement_counts["Not Placed (%)"] = (project_placement_counts[0] / project_placement_counts["Total"]) * 100
project_placement_counts["Placed (%)"] = (project_placement_counts[1] / project_placement_counts["Total"]) * 100

# Xuất bảng thống kê
print("\nBảng thống kê số lượng và tỷ lệ sinh viên theo số lượng dự án:")
print(project_placement_counts)

#Vẽ biểu đồ
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Biểu đồ 1: Tổng số sinh viên theo số lượng dự án
sns.countplot(x="Projects", data=df, color="blue", ax=axes[0])
axes[0].set_xlabel("Số lượng dự án")
axes[0].set_ylabel("Số lượng sinh viên")
axes[0].set_title("Tổng số sinh viên theo số lượng dự án")

# Biểu đồ 2: Số lượng sinh viên được tuyển và không được tuyển theo số dự án
df_placement = df.groupby(["Projects", "PlacementStatus"]).size().unstack().fillna(0)
df_placement.plot(kind="bar", stacked=True, color=["blue", "pink"], ax=axes[1])
axes[1].set_xlabel("Số lượng dự án")
axes[1].set_ylabel("Số lượng sinh viên")
axes[1].set_title("Số lượng sinh viên được tuyển và không được tuyển")
axes[1].legend(["Không được tuyển", "Được tuyển"])

# Biểu đồ 3: Tỷ lệ phần trăm sinh viên được tuyển theo số lượng dự án
df_percent = df_placement.div(df_placement.sum(axis=1), axis=0) * 100
df_percent.plot(kind="bar", stacked=True, color=["blue", "pink"], ax=axes[2])
axes[2].set_xlabel("Số lượng dự án")
axes[2].set_ylabel("Tỷ lệ (%)")
axes[2].set_title("Tỷ lệ phần trăm sinh viên được tuyển và không được tuyển")
axes[2].legend(["Không được tuyển", "Được tuyển"])

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
