# %% Import thư viện
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# %% Đọc dữ liệu
file_path = "D:/TAI LIEU/ĐỒ ÁN/NỘP FINAL/MÃ NGUỒN, DATASET/BÀI TOÁN 2/0.DATASET/processed_data.csv"
df = pd.read_csv(file_path)

# %% Xác định biến phụ thuộc
y = df["PlacementStatus"]

# %% Phân loại biến độc lập
binary_vars = ["ExtracurricularActivities"]
ordinal_vars = ["CGPA", "Projects", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]

# %% Hồi quy logistic đơn biến với biến nhị phân và vẽ biểu đồ
logit_results_binary = {}
for var in binary_vars:
    X_binary = sm.add_constant(df[var])  # Thêm hệ số chặn
    logit_model_binary = sm.Logit(y, X_binary).fit(disp=0)
    logit_results_binary[var] = logit_model_binary.summary()
    print(f"\nHồi quy với biến nhị phân {var}:")
    print(logit_model_binary.summary())

    # Vẽ biểu đồ cột
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[var], hue=y, palette=["blue", "orange"])
    plt.title(f"Phân phối {var} theo PlacementStatus")
    plt.xlabel(var)
    plt.ylabel("Số lượng")
    plt.legend(title="PlacementStatus", labels=["Không tuyển", "Được tuyển"])
    plt.show()

# %% Hồi quy logistic đơn biến với biến thứ bậc và vẽ biểu đồ
logit_results_ordinal = {}
for var in ordinal_vars:
    X_ordinal = sm.add_constant(df[var])  # Thêm hệ số chặn
    logit_model_ordinal = sm.Logit(y, X_ordinal).fit(disp=0)
    logit_results_ordinal[var] = logit_model_ordinal.summary()
    print(f"\nHồi quy với biến thứ bậc {var}:")
    print(logit_model_ordinal.summary())

    # Vẽ biểu đồ violin
    plt.figure(figsize=(6, 4))
    sns.violinplot(x=y, y=df[var], palette=["blue", "orange"])
    plt.title(f"Phân phối {var} theo PlacementStatus")
    plt.xlabel("PlacementStatus (0 = Không tuyển, 1 = Được tuyển)")
    plt.ylabel(var)
    plt.show()