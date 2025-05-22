pip install mlxtend

  ## Kết nối Google Drive
from google.colab import drive
import pandas as pd

drive.mount("/content/gdrive", force_remount = True)

folder = '/content/gdrive/My Drive/ML_Final Project/'

print(f"Số lượng dòng: {df.shape[0]}")
print(f"Số lượng cột: {df.shape[1]}")

df.info()

print(df.nunique().sort_values(ascending=False))

df = df.drop(columns=["Invoice_ID", "Basket_ID"])

# Chuyển đổi cột ngày về dạng datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Last_Purchase_Date'] = pd.to_datetime(df['Last_Purchase_Date'], errors='coerce')

# Chuyển các cột nhị phân mang tính phân loại sang kiểu object
binary_cols = ['Is_Member', 'Fraud_Suspicion']
df[binary_cols] = df[binary_cols].astype('object')

missing_values = df.isnull().sum()
print("Số lượng giá trị thiếu theo từng cột:\n", missing_values)

# Áp dụng quy tắc 3-sigma để phát hiện và xử lý giá trị nhiễu
def check_outliers_using_3_sigma(df, verbose=True):
    has_outliers = True
    while has_outliers:
        has_outliers = False
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                mean = df[col].mean()
                std = df[col].std(ddof=1)
                upper = mean + 3 * std
                lower = mean - 3 * std
                outliers = df[(df[col] > upper) | (df[col] < lower)]

                if not outliers.empty:
                    if verbose:
                        print(f"Cột {col} có {outliers.shape[0]} giá trị nhiễu nằm ngoài khoảng ({lower:.2f}, {upper:.2f})")
                    df.loc[(df[col] > upper) | (df[col] < lower), col] = df[col].median()
                    has_outliers = True
                else:
                    if verbose:
                        print(f"Cột {col}: Không có outlier.")
check_outliers_using_3_sigma(df)

# Rời rạc hóa các biến định lượng thành các nhóm (theo quantile)
continuous_cols = ['Total_Spending (Last 3 Months)', 'Avg_Monthly_Spending']

for col in continuous_cols:
    new_col = col + '_Group'
    df[new_col] = pd.qcut(df[col], q=4, labels=['Thấp', 'Trung bình', 'Cao', 'Rất cao'])

# Chuyển toàn bộ các cột sang object (nếu cần OneHotEncoding sau này)
df = df.astype({col: 'object' for col in df.columns if df[col].dtype not in ['datetime64[ns]']})

# Kiểm tra lại kiểu dữ liệu sau khi xử lý
df.info()


# // TÌM CÁC MẪU PHỔ BIẾN BẰNG FP-GROWTH

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# tạo cột Month
df['Month'] = df['Date'].dt.to_period('M')
# Gom các sản phẩm theo Invoice_ID theo từng tháng
grouped_monthly = df.groupby(['Month', 'Invoice_ID'])['Product'].apply(list).reset_index()

fp_results_by_month = {}

# Áp dụng FP-Growth cho từng tháng
for month, group in grouped_monthly.groupby('Month'):
    transactions = group['Product'].tolist()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)
    frequent_itemsets['Month'] = str(month)

    fp_results_by_month[str(month)] = frequent_itemsets

# Gộp kết quả lại để tiện theo dõi
all_fp_items = pd.concat(fp_results_by_month.values(), ignore_index=True)
all_fp_items = all_fp_items.sort_values(by=['Month', 'support'], ascending=[True, False])

month = '2024-01'
# Hiển thị một số kết quả
top10_selected_month = all_fp_items[all_fp_items['Month'] == month].sort_values(by='support', ascending=False).head(10)

print(f"Top 10 mẫu phổ biến trong tháng {month}:")
print(top10_selected_month)


# // DỰ ĐOÁN CÁC SẢN PHẨM NÊN NHẬP BẰNG SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Tổng số lượng nhập mỗi tháng mỗi sản phẩm
monthly_data = df.groupby(['Product', 'Month'])['Quantity'].sum().reset_index()

# Sắp xếp theo Product và Month
monthly_data = monthly_data.sort_values(by=['Product', 'Month'])

# Tạo biến mục tiêu: có nhập ở tháng sau không
monthly_data['Nhap_Thang_Sau'] = monthly_data.groupby('Product')['Quantity'].shift(-1)
monthly_data['Nhap_Thang_Sau'] = (monthly_data['Nhap_Thang_Sau'] > 0).astype(int)

# Loại bỏ các hàng cuối mỗi chuỗi sản phẩm (vì không có tháng sau)
monthly_data = monthly_data.dropna()

# Lọc ra các sản phẩm có cả tháng nhập và không nhập
valid_products = monthly_data.groupby('Product')['Nhap_Thang_Sau'].nunique()
valid_products = valid_products[valid_products == 2].index
monthly_data = monthly_data[monthly_data['Product'].isin(valid_products)]

# === Biến đặc trưng và mục tiêu ===
X = monthly_data[['Product', 'Quantity']]  # có thể thêm Month nếu muốn
y = monthly_data['Nhap_Thang_Sau']

# Mã hóa biến phân loại
X['Product'] = LabelEncoder().fit_transform(X['Product'])

# Tách train/test (đảm bảo giữ tỷ lệ 0/1 giống nhau)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Kiểm tra lại phân phối lớp
print("Phân phối lớp trong train:", y_train.value_counts())
print("Phân phối lớp trong test:", y_test.value_counts())

# Huấn luyện mô hình SVM
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# // DỰ ĐOÁN SỐ LƯỢNG SẢN PHẨM CẦN NHẬP BẰNG LINEAR REGRESSION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Thêm cột thể hiện tháng từ cột ngày giao dịch
df['Month'] = df['Date'].dt.to_period('M')

# Tính tổng số lượng sản phẩm bán được theo từng tháng
monthly_data = df.groupby(['Product', 'Month'])['Quantity'].sum().reset_index()

# Sắp xếp dữ liệu theo tên sản phẩm và theo thứ tự thời gian
monthly_data = monthly_data.sort_values(by=['Product', 'Month'])

# Tạo cột thể hiện số lượng bán của tháng kế tiếp
monthly_data['Next_Quantity'] = monthly_data.groupby('Product')['Quantity'].shift(-1)

# Loại bỏ các dòng không có dữ liệu tháng kế tiếp
monthly_data = monthly_data.dropna()

# Mã hóa tên sản phẩm thành số để mô hình có thể học được
le = LabelEncoder()
monthly_data['Product_encoded'] = le.fit_transform(monthly_data['Product'])

# Chọn cột đặc trưng và cột mục tiêu để đưa vào mô hình
X = monthly_data[['Product_encoded', 'Quantity']]
y = monthly_data['Next_Quantity']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra và tính sai số
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")

# Tạo bảng kết quả để so sánh thực tế và dự đoán
result_df = X_test.copy()
result_df['Actual_Quantity_Next_Month'] = y_test.values
result_df['Predicted_Quantity_Next_Month'] = y_pred

# Giải mã lại tên sản phẩm từ mã số
result_df['Product'] = le.inverse_transform(result_df['Product_encoded'])

# Tổng hợp theo sản phẩm để lấy số lượng gần nhất và trung bình dự đoán
summary = result_df.groupby('Product').agg({
    'Quantity': 'last',
    'Predicted_Quantity_Next_Month': 'mean'
}).reset_index()

# Làm tròn giá trị dự đoán
summary['Predicted_Quantity_Next_Month'] = summary['Predicted_Quantity_Next_Month'].round(2)

# Xác định sản phẩm nào cần nhập thêm
summary['Should_Import'] = summary['Predicted_Quantity_Next_Month'] > 0

# Gợi ý số lượng nên nhập thêm, tăng 20 phần trăm so với dự đoán
summary['Suggested_Import_Quantity'] = np.where(summary['Should_Import'],np.ceil(summary['Predicted_Quantity_Next_Month'] * 1.2),0).astype(int)

# In ra bảng kết quả cuối cùng
print("\n=== Dự đoán & Gợi ý nhập hàng ===")
print(summary[['Product', 'Quantity', 'Predicted_Quantity_Next_Month', 'Should_Import', 'Suggested_Import_Quantity']])


# // BIỂU DIỄN BIỂU ĐỒ TRỰC QUAN
import pandas as pd
import matplotlib.pyplot as plt

# Tạo cột "Month" với định dạng YYYY-MM
df['Month'] = df['Date'].dt.to_period('M').astype(str)

# Gom nhóm theo sản phẩm và tháng, tính tổng số lượng bán
monthly_sales = df.groupby(['Product', 'Month'])['Quantity'].sum().reset_index()

# Danh sách sản phẩm duy nhất
products = monthly_sales['Product'].unique()

# Vẽ biểu đồ đường cho từng sản phẩm
plt.figure(figsize=(14, 7))

for product in products:
    product_data = monthly_sales[monthly_sales['Product'] == product]
    plt.plot(product_data['Month'], product_data['Quantity'], marker='o', label=product)

plt.title('Số lượng sản phẩm bán được theo tháng')
plt.xlabel('Tháng')
plt.ylabel('Số lượng bán')
plt.xticks(rotation=45)
plt.legend(title='Sản phẩm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Tạo các cột 'Year' và 'Month' từ cột 'Date'
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Xác định năm gần nhất có dữ liệu để dùng làm "năm trước"
last_year = df['Year'].max()
growth_rate = 1.2  # hệ số tăng trưởng giả định

# Nhóm dữ liệu theo Sản phẩm và Tháng trong năm trước
sales_last_year = df[df['Year'] == last_year].groupby(['Product', 'Month'])['Quantity'].sum().reset_index()

# Nếu không có dữ liệu năm trước, in thông báo
if sales_last_year.empty:
    print(f"⚠️ Không có dữ liệu cho năm {last_year} để dự báo.")
else:
    # Dự đoán số lượng cần nhập cho năm tới
    sales_last_year['Suggested_Import'] = sales_last_year['Quantity'] * growth_rate

    # Lấy danh sách các sản phẩm
    products = sales_last_year['Product'].unique()
    num_products = len(products)

    # Tạo grid biểu đồ
    fig, axes = plt.subplots(nrows=(num_products + 1) // 2, ncols=2, figsize=(16, max(4, num_products * 2.5)))
    axes = axes.flatten()

    # Vẽ biểu đồ cho từng sản phẩm
    for i, product in enumerate(products):
        data = sales_last_year[sales_last_year['Product'] == product]
        sns.barplot(data=data, x='Month', y='Suggested_Import', ax=axes[i], palette='Blues_d')
        axes[i].set_title(f'Sản phẩm: {product}')
        axes[i].set_ylabel('SL Nhập Dự Kiến')
        axes[i].set_xlabel('Tháng')

    # Xóa các biểu đồ thừa nếu số sản phẩm lẻ
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Chỉnh layout và thêm tiêu đề lớn
    plt.tight_layout()
    plt.suptitle('Hình 6.2 - Biểu đồ đề xuất số lượng nhập hàng theo sản phẩm và tháng\n(Dựa trên dữ liệu bán hàng năm trước và hệ số tăng trưởng 1.2)', fontsize=16, y=1.03)
    plt.show()
