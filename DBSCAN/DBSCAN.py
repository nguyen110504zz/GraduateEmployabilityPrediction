#%% Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#%% Đọc dữ liệu
data = pd.read_csv("D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv")

#%% Loại bỏ biến phân loại
label = data["PlacementStatus"]
data = data.drop(columns=["PlacementStatus"])

#%% Lấy dữ liệu số để phân cụm
numerical_data = data.select_dtypes(include=[np.number])

#%% Tìm giá trị tối ưu cho eps và min_samples
eps_values = np.arange(0.1, 0.9, 0.1)  # Từ 0.1 đến 1.0, bước nhảy 0.1
min_samples_values = range(2, 10)  # Từ 2 đến 10

best_score = -1
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(numerical_data)

        # Bỏ qua trường hợp chỉ có một cụm hoặc tất cả là nhiễu
        if len(set(labels)) <= 1:
            continue

        score = silhouette_score(numerical_data, labels)

        if score > best_score:
            best_score = score
            best_eps = eps
            best_min_samples = min_samples

print(f"Thông số tối ưu: eps = {best_eps}, min_samples = {best_min_samples}")
print(f"Silhouette Score cao nhất: {best_score}")

#%% Áp dụng DBSCAN với thông số tối ưu
optimal_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
optimal_labels = optimal_dbscan.fit_predict(numerical_data)

#%% Giảm chiều dữ liệu bằng PCA để vẽ biểu đồ
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(numerical_data)

#%% Vẽ kết quả phân cụm DBSCAN
plt.figure(figsize=(6, 5))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=optimal_labels, cmap='rainbow', edgecolors='k')
plt.title("DBSCAN Clustering (Optimized)")
plt.show()

#%% Đánh giá kết quả DBSCAN
df_results = pd.DataFrame({"DBSCAN_Cluster": optimal_labels})
if label is not None:
    df_results["PlacementStatus"] = label

    # Đếm số lượng sinh viên trong từng cụm và tỷ lệ trúng tuyển
    dbscan_stats = df_results.groupby("DBSCAN_Cluster")["PlacementStatus"].agg(['count', 'mean'])
    dbscan_stats.columns = ["Số lượng sinh viên", "Tỷ lệ trúng tuyển"]

    print("\nThống kê số lượng sinh viên và tỷ lệ trúng tuyển trong từng cụm (DBSCAN):")
    print(dbscan_stats)
