import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense

data = pd.read_csv('your_data.csv')
# 删除包含缺失值的行
data = data.dropna()

# 使用平均值填充缺失值
data = data.fillna(data.mean())
# 使用 Z-score 进行异常值检测和处理
from scipy.stats import zscore

z_scores = zscore(data.iloc[:, 1:])  # 排除'user_id'列
data = data[(np.abs(z_scores) < 3).all(axis=1)]  # 保留 Z-score 小于3的数据
# 去除重复行
data = data.drop_duplicates()
# 示例：创建新特征 '总消费'，是 '上网时间' 与 '消费' 的乘积
data['总消费'] = data['上网时间'] * data['消费']

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, 1:])  # 排除'user_id'列

# 划分训练集和测试集
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# 构建多输出的自编码器模型
input_dim = X_train.shape[1]
encoding_dim = 3  # 选择一个适当的编码维度

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)

# 添加六个输出层，对应六个维度的评分
output_layers = []
for i in range(6):
    output_layers.append(Dense(1, activation='linear')(encoder))

autoencoder = Model(inputs=input_layer, outputs=output_layers)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 训练多输出的自编码器模型
autoencoder.fit(X_train, [X_train[:, i] for i in range(6)], epochs=50, batch_size=32, shuffle=True,
                validation_data=(X_test, [X_test[:, i] for i in range(6)]))

# 使用训练好的自编码器模型进行评分预测
user_id = 123
user_data = data[data['user_id'] == user_id].iloc[:, 1:]
user_data_scaled = scaler.transform(user_data)

predicted_scores = autoencoder.predict(user_data_scaled.reshape(1, -1))

# 创建包含预测评分的DataFrame
predicted_scores_df = pd.DataFrame(columns=['维度', '预测评分'])
for i, dimension in enumerate(['上网时间', '消费', '日常行为', '关注信息', '社交媒体']):
    predicted_scores_df = predicted_scores_df.append({'维度': dimension, '预测评分': predicted_scores[i][0]}, ignore_index=True)

# 输出预测评分
print("预测评分:")
print(predicted_scores_df)

# 将预测评分保存到本地文件夹
predicted_scores_df.to_csv('predicted_scores.csv', index=False)