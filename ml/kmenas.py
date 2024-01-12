import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 假设你有一个包含电商用户数据的DataFrame，具体列名可以根据你的实际数据进行调整
# 假设列名为：'user_id', 'recency', 'frequency', 'monetary'
# 你需要根据实际情况调整列名
data = pd.read_csv('data.csv')

# 提取RFM特征
rfm_data = data[['recency', 'frequency', 'monetary']]

# 数据标准化
scaler = StandardScaler()
rfm_data_scaled = scaler.fit_transform(rfm_data)

# 使用K均值聚类
kmeans = KMeans(n_clusters=3, random_state=42)  # 假设分为3个群
data['cluster'] = kmeans.fit_predict(rfm_data_scaled)

# 绘制聚类结果的散点图
plt.scatter(rfm_data_scaled[:, 0], rfm_data_scaled[:, 1], c=data['cluster'], cmap='viridis')
plt.title('K均值聚类结果')
plt.xlabel('Recency (标准化)')
plt.ylabel('Frequency (标准化)')
plt.show()

# 打印每个群的统计信息
cluster_stats = data.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean',
    'user_id': 'count'
}).rename(columns={'user_id': 'count'})
print("每个群的统计信息:")
print(cluster_stats)
result_dir = 'cluster_results'
os.makedirs(result_dir, exist_ok=True)
result_file_path = os.path.join(result_dir, 'data1.ata.csv')
data.to_csv(result_file_path, index=False)