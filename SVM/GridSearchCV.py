#%% Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
#%% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv"
df = pd.read_csv(file_path)

#%% Tách biến độc lập và biến mục tiêu
X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

#%% Chia dữ liệu thành tập train (80%) và test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%% Khởi tạo GridSearchCV để tối ưu hóa tham số của SVM
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}
#%% Huấn luyện mô hình với GridSearchCV
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

#%% Lấy mô hình tối ưu nhất
best_model = grid_search.best_estimator_
print(best_model)
#%%
print("Best kernel:", best_model.kernel)
