#%% Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#%% Đọc dữ liệu
data = pd.read_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv")

#%% Lưu lại PlacementStatus trước khi xóa
placement_labels = data["PlacementStatus"]
data = data.drop(columns=['PlacementStatus'])

#%% Lấy dữ liệu số
numerical_data = data.select_dtypes(include=[np.number]).dropna()

#%% K-Means
wcss = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(numerical_data)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(numerical_data, labels))

#%% Silhouette Score
plt.figure(figsize=(6, 4))
plt.plot(k_values, silhouette_scores, marker='s', linestyle='--')
plt.xlabel('Số cụm K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.show()

#%% Chọn số cụm tối ưu
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(numerical_data)

#%% Giảm chiều để trực quan hóa
if numerical_data.shape[1] >= 2:  # Kiểm tra đủ chiều cho PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(numerical_data)

    plt.figure(figsize=(6, 5))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')
    plt.title("K-Means Clustering")
    plt.show()
else:
    print("Không thể thực hiện PCA vì số lượng đặc trưng quá ít!")

#%% Đánh giá kết quả
df_results = pd.DataFrame({"KMeans_Cluster": kmeans_labels, "PlacementStatus": placement_labels})

#%% Kiểm tra kết quả phân cụm
print("\nTỷ lệ sinh viên trúng tuyển trong từng cụm (K-Means):")
print(df_results.groupby("KMeans_Cluster")["PlacementStatus"].mean())

#%% Thống kê số lượng sinh viên và tỷ lệ trúng tuyển
cluster_stats = df_results.groupby("KMeans_Cluster")["PlacementStatus"].agg(['count', 'mean'])
cluster_stats.columns = ["Số lượng sinh viên", "Tỷ lệ trúng tuyển"]
print("\nThống kê số lượng sinh viên và tỷ lệ trúng tuyển trong từng cụm:")
print(cluster_stats)

#%% Tính toán đặc điểm trung bình của từng cụm
cluster_characteristics = numerical_data.copy()
cluster_characteristics["Cluster"] = kmeans_labels
cluster_characteristics = cluster_characteristics.groupby("Cluster").mean()

#%% Hiển thị đặc điểm trung bình của từng cụm
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\nĐặc điểm trung bình của từng cụm:")
print(cluster_characteristics)
