#%% Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
#%% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv"
df = pd.read_csv(file_path)

#%% Tách biến độc lập và biến mục tiêu
X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

#%% Áp dụng K-Fold Cross Validation (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#%% Áp dụng PCA để giảm số chiều xuống 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#%% Khởi tạo mô hình SVM với kernel RBF
svm_rbf = SVC(kernel="rbf", C=10, gamma=1, random_state=42)

#%% Thực hiện K-Fold Cross Validation và tính độ chính xác
accuracy_scores = cross_val_score(svm_rbf, X_pca, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(svm_rbf, X_pca, y, cv=kf, scoring='precision')
recall_scores = cross_val_score(svm_rbf, X_pca, y, cv=kf, scoring='recall')
f1_scores = cross_val_score(svm_rbf, X_pca, y, cv=kf, scoring='f1')

#%% Tạo DataFrame chứa kết quả từ các Fold
df_results = pd.DataFrame({
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1-Score': f1_scores
})

#%% Tính trung bình và độ lệch chuẩn từng cột
mean_values = df_results.mean()
std_values = df_results.std()

#%% Thêm dòng trung bình và độ lệch chuẩn vào DataFrame
df_results.loc['Mean'] = mean_values
df_results.loc['Std'] = std_values

#%% Hiển thị kết quả
print("DataFrame kết quả từng Fold:")
print(df_results)
print("\nTrung bình từng chỉ số:")
print(mean_values)
