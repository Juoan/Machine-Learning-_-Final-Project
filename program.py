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
from collections import defaultdict

class ItemsAndOrderedItems:
    def __init__(self, transaction_id):
        self.transaction_id = transaction_id
        self.items_list = []
        self.ordered_items_list = []

    def add_item(self, item):
        self.items_list.append(item)

    def create_ordered_items(self, fp_set_list):
        for item in fp_set_list:
            if item in self.items_list:
                self.ordered_items_list.append(item)

class TransactionIDs:
    def __init__(self):
        self.transaction_ids_dic = {}
        self.items_with_frequency_count = defaultdict(int)
        self.frequent_pattern_set = {}

    def add_transaction_id_with_items(self, tid, items):
        if tid not in self.transaction_ids_dic:
            self.transaction_ids_dic[tid] = ItemsAndOrderedItems(tid)

        for item in items:
            self.transaction_ids_dic[tid].add_item(item)
            self.items_with_frequency_count[item] += 1

    def create_frequent_pattern_set_with_frequency_greater_than(self, count):
        ordered_items = []

        for item, freq in self.items_with_frequency_count.items():
            if freq >= count:
                self._insert_ordered(item, ordered_items)

        for item in ordered_items:
            self.frequent_pattern_set[item] = self.items_with_frequency_count[item]

    def _insert_ordered(self, item, ordered_items):
        for i in range(len(ordered_items) + 1):
            if i == len(ordered_items):
                ordered_items.append(item)
                break
            elif self.items_with_frequency_count[item] >= self.items_with_frequency_count[ordered_items[i]]:
                ordered_items.insert(i, item)
                break

    def initialize_ordered_item_set_for_each_transaction(self):
        for trans in self.transaction_ids_dic.values():
            trans.create_ordered_items(self.frequent_pattern_set)


class FPGraphItemNode:
    def __init__(self, number_id, name):
        self.item_node_name = name
        self.item_node_number_id = number_id
        self.item_node_id = f"{name}_WithNumberID_{number_id}"
        self.count = 1
        self.parent_node_id = ""
        self.end_item_nodes = {}
        self.linked_item_node_ids = []

    def add_parent_node(self, parent_node_id):
        self.parent_node_id = parent_node_id

    def add_end_item_node(self, name, number_id):
        self.end_item_nodes[name] = number_id

    def add_linked_item_node_id(self, node_id):
        self.linked_item_node_ids.append(node_id)

    def update_count(self):
        self.count += 1


class FPGraph:
    def __init__(self, transaction_ids):
        self.item_node_ids = {}
        self.current_bait_node_ids = {}
        self.add_item_node("Null", 1)
        self.current_bait_node_ids["Null"] = 1
        self.building_fp_tree(transaction_ids)

    def building_fp_tree(self, transaction_ids):
        for trans in transaction_ids.transaction_ids_dic.values():
            current_node_id = "Null_WithNumberID_1"
            for i, item in enumerate(trans.ordered_items_list):
                if item not in self.item_node_ids[current_node_id].end_item_nodes:
                    number_id = self.current_bait_node_ids.get(item, 0) + 1
                    self.add_item_node(item, number_id)
                    self.add_directed_edge(current_node_id, item, number_id)
                    self.current_bait_node_ids[item] = number_id
                    self.insert_set_to_tree(i, trans.ordered_items_list)
                    break
                else:
                    end_id = f"{item}_WithNumberID_{self.item_node_ids[current_node_id].end_item_nodes[item]}"
                    self.update_count_to_item_node_id(end_id)
                    current_node_id = end_id

    def insert_set_to_tree(self, index, ordered_items):
        for i in range(index, len(ordered_items) - 1):
            current = ordered_items[i]
            next_item = ordered_items[i + 1]
            next_id = self.current_bait_node_ids.get(next_item, 0) + 1

            self.add_item_node(next_item, next_id)
            self.add_directed_edge(f"{current}_WithNumberID_{self.current_bait_node_ids[current]}", next_item, next_id)

            if next_item in self.current_bait_node_ids:
                self.add_linked_edge(f"{next_item}_WithNumberID_{self.current_bait_node_ids[next_item]}",
                                     f"{next_item}_WithNumberID_{next_id}")

            self.current_bait_node_ids[next_item] = next_id

    def add_item_node(self, name, number_id):
        node_id = f"{name}_WithNumberID_{number_id}"
        if node_id not in self.item_node_ids:
            self.item_node_ids[node_id] = FPGraphItemNode(number_id, name)

    def add_directed_edge(self, source_id, name, number_id):
        self.item_node_ids[source_id].add_end_item_node(name, number_id)
        end_id = f"{name}_WithNumberID_{number_id}"
        self.item_node_ids[end_id].add_parent_node(source_id)

    def add_linked_edge(self, first_id, second_id):
        if second_id not in self.item_node_ids[first_id].linked_item_node_ids:
            self.item_node_ids[first_id].add_linked_item_node_id(second_id)
        if first_id not in self.item_node_ids[second_id].linked_item_node_ids:
            self.item_node_ids[second_id].add_linked_item_node_id(first_id)

    def update_count_to_item_node_id(self, node_id):
        self.item_node_ids[node_id].update_count()


class ItemWithCPBAndFPTree:
    def __init__(self, item_name, item_node_ids):
        self.item_name = item_name
        self.list_of_cpb = []
        self.items_with_count = defaultdict(int)
        self.cpft_node_names = []
        self.cpft_node_name_count = 0
        self._add_to_list_of_cpb(item_name, item_node_ids)
        self._find_cpft()

    def _add_to_list_of_cpb(self, item_name, item_node_ids):
        for node in item_node_ids.values():
            if node.item_node_name == item_name:
                self._trace_and_add(node.item_node_id, item_node_ids)

    def _trace_and_add(self, node_id, item_node_ids):
        path = []
        while node_id != "Null_WithNumberID_1":
            name = item_node_ids[node_id].item_node_name
            path.append(name)
            self.items_with_count[name] += 1
            node_id = item_node_ids[node_id].parent_node_id
        self.list_of_cpb.append(path)

    def _find_cpft(self):
        for name, count in self.items_with_count.items():
            if count > self.cpft_node_name_count:
                self.cpft_node_names = [name]
                self.cpft_node_name_count = count
            elif count == self.cpft_node_name_count:
                self.cpft_node_names.append(name)
  df['Month'] = df['Date'].dt.to_period('M')  # chuyển ngày về tháng
monthly_data = df.groupby(['Product', 'Month']).agg({
    'Quantity': 'sum'
}).reset_index()

# Tạo biến mục tiêu: nếu Quantity > 0 thì có nhập
monthly_data['Nhap'] = (monthly_data['Quantity'] > 0).astype(int)


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
