import pandas as pd

# 读取CSV文件
df = pd.read_csv('UserBehavior.csv', header=None)

# 添加列名
df.columns = ["User_ID", "Item_ID", "CateGory_ID", "Behavior_type", "Timestamp"] + list(df.columns[5:])

# 将DataFrame写回CSV文件
df.to_csv('UserBehavior_with_header.csv', index=False, header=True)

# 展示修改后的DataFrame
print(df.head())
