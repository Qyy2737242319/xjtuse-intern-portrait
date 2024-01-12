import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_csv('data.csv',encoding = 'ISO-8859-1', dtype = {'CustomerID':str})
print(df.shape)

print(df.apply(lambda x :sum(x.isnull())/len(x),axis=0))

df.drop(['Description'],axis=1,inplace=True)
df.head()
df['CustomerID'] = df['CustomerID'].astype('str')
df.info()
df['CustomerID'] = df['CustomerID'].fillna('unknown')
print(df)
df['date'] = [x.split(' ')[0] for x in df['InvoiceDate']]
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
df[['date', 'month']]
df = df.drop_duplicates()
df.shape
df[(df['Quantity']<0) | (df['UnitPrice']<0)]
df = df[(df['Quantity']>0) & (df['UnitPrice']>0)]
df[(df['Quantity']<0) | (df['UnitPrice']<0)]
R_value = df.groupby('CustomerID')['date'].max()
R_value = (df['date'].max() - R_value).dt.days  # 这里将2011-12-9作为当前日期进行计算
print(R_value)
F_value = df.groupby('CustomerID')['InvoiceNo'].nunique()
F_value
# 首先计算每个订单的消费金额
df['amount'] = df['Quantity'] * df['UnitPrice']
# 再计算M值
M_value = df.groupby('CustomerID')['InvoiceNo'].nunique()
M_value = df.groupby('CustomerID')['amount'].sum()
M_value
R_value.describe()
R_value.hist(bins = 30)
R_value.hist(bins = 30)
M_value.describe()
M_value.hist(bins = 30)
M_value.plot.box()
M_value[M_value<2000].hist(bins = 30)
F_value.quantile([0.1,0.2,0.3,0.4,0.5,0.9,1])
F_value.hist(bins = 30)
F_value[F_value<30].hist(bins = 30)
#五级
R_bins = [0,30,90,180,360,720]
F_bins = [1,2,5,10,20,5000]
M_bins = [0,500,2000,5000,10000,200000]
R_score = pd.cut(R_value,R_bins,labels=[5,4,3,2,1],right=False)
print("----------------------------------")
print(R_score)
F_score = pd.cut(F_value,F_bins,labels=[1,2,3,4,5],right=False)
F_score
M_score = pd.cut(M_value,M_bins,labels=[1,2,3,4,5],right=False)
M_score
rfm = pd.concat([R_score,F_score,M_score],axis=1)
rfm.rename(columns={'date':'R_score','InvoiceNo':'F_score','amount':'M_score'},inplace=True)
print(rfm)
rfm['R_score'] = rfm['R_score'].astype('float')
rfm['F_score'] = rfm['F_score'].astype('float')
rfm['M_score'] = rfm['M_score'].astype('float')
rfm.describe()
rfm['R'] = np.where(rfm['R_score']>3.82,'高','低')
rfm['F'] = np.where(rfm['F_score']>2.03,'高','低')
rfm['M'] = np.where(rfm['M_score']>1.89,'高','低')
rfm
rfm['RFM']=rfm['R']+rfm['F']+rfm['M']
print(rfm)


def rfm2grade(x):
    if x == '高高高':
        return '高价值客户'
    elif x == '高低高':
        return '重点发展客户'
    elif x == '低高高':
        return '重点保持客户'
    elif x == '低低高':
        return '重点挽留客户'
    elif x == '高高低':
        return '一般价值客户'
    elif x == '高低低':
        return '一般发展客户'
    elif x == '低高低':
        return '一般保持客户'
    else:
        return '一般挽留客户'


rfm['用户等级'] = rfm['RFM'].apply(rfm2grade)
print(rfm)
rfm['id'] = range(1, len(rfm) + 1)
output_csv_path = 'rfm.csv'

# Export the DataFrame to the CSV file
rfm.to_csv(output_csv_path, index=False)

print(f'DataFrame exported to {output_csv_path}')
# from sqlalchemy import create_engine
#
# # Assuming 'rfm' is the DataFrame you want to write to the database
#
# # MySQL database connection parameters
# db_username = 'root'
# db_password = 'admin'
# db_host = '192.168.52.205'
# db_port = '3306'
# db_name = 'tags_dat'
#
# # Define the database connection URL for MySQL
# database_url = f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
#
# # Create a database engine
# engine = create_engine(database_url)
#
# # Write the DataFrame to the MySQL database
# rfm.to_sql('rfm_data_table', engine, index=False, if_exists='replace')
#
# # Close the database connection
# engine.dispose()
