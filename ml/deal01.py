from pyspark.sql import SparkSession
from pyspark import SparkContext, StorageLevel
from pyspark.sql.functions import *
import pandas
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'simhei'
spark = SparkSession.builder.getOrCreate()

# 从mysql上读取数据
url = "jdbc:mysql://localhost:3306/taobao_data"
propertie = {
    "user": "root",
    "password": "mysql123",
    "dirver": "com.mysql.cj.jdbc.Driver"
}
user_retention_data = spark.read.jdbc(url=url, table="user_retention_data", properties=propertie)
date_flow = spark.read.jdbc(url=url, table="date_flow", properties=propertie)
month_flow = spark.read.jdbc(url=url, table="month_flow", properties=propertie)
click_data = spark.read.jdbc(url=url, table="click_data", properties=propertie)

# 统计7日留存率， 14日留存率， 28日留存率
all_user_num = user_retention_data.count()
seven_retention = user_retention_data.filter(user_retention_data.retention_date >= 7).count() / all_user_num
fourteen_retention = user_retention_data.filter(user_retention_data.retention_date >= 14).count() / all_user_num
te_retention = user_retention_data.filter(user_retention_data.retention_date >= 28).count() / all_user_num

retention_y = [seven_retention, fourteen_retention, te_retention]
retention_x = ["7日留存率", "14日留存率", "28日留存率"]

plt.plot(retention_x, retention_y, color='r', linewidth=2, linestyle='dashdot')

for x in range(3):
    plt.text(x - 0.13, retention_y[x], str(retention_y[x]), ha='center', va='bottom', fontsize=9)

plt.savefig("retention.jpg")
plt.clf()

# 统计日均流量
date_flow = date_flow.sort("hour").toPandas()
date_flow_x = date_flow["hour"].values
date_flow_y = date_flow["count"].values

plt.figure(figsize=(8, 4))
plt.plot(date_flow_x, date_flow_y, color='r', linewidth=2, linestyle='dashdot')

for x in range(24):
    plt.text(x - 0.13, date_flow_y[x], str(date_flow_y[x]), ha='center', va='bottom', fontsize=9)

plt.savefig("date_flow.jpg")
plt.clf()

# 统计月均流量
month_flow = month_flow.sort("day").toPandas()
month_flow_x = month_flow["day"].values
month_flow_y = month_flow["count"].values

plt.figure(figsize=(15, 4))
plt.xticks(rotation=90)
plt.plot(month_flow_x, month_flow_y, color='r', linewidth=2, linestyle='dashdot')
plt.savefig("month_flow.jpg", bbox_inches='tight')
plt.clf()


# 统计top10的商品
def take(x):
    data_list = []
    for i in range(10):
        data_list.append(x[i])
    return data_list


visit_data = click_data.sort(desc("clike_num")).toPandas()
visit_x = take(visit_data["item_category"].values)
visit_y = take(visit_data["clike_num"].values)

visit_plt = plt.bar(visit_x, visit_y, lw=0.5, fc="b", width=0.5)
plt.bar_label(visit_plt, label_type='edge')
plt.savefig("visit_top10.jpg", bbox_inches='tight')
plt.clf()

buy_data = click_data.sort(desc("buy_num")).toPandas()
buy_x = take(buy_data["item_category"].values)
buy_y = take(buy_data["buy_num"].values)

buy_plt = plt.bar(buy_x, buy_y, lw=0.5, fc="b", width=0.5)
plt.bar_label(buy_plt, label_type='edge')
plt.savefig("buy_top10.jpg", bbox_inches='tight')
plt.clf()

# 统计购买路径
buy_path_data = user_retention_data.filter(user_retention_data.buy_path != "没有购买").groupBy("buy_path").count().sort(
    desc("count")).toPandas()
buy_path_y = buy_path_data["count"].values
buy_path_x = buy_path_data["buy_path"].values

buy_path_plt = plt.bar(buy_path_x, buy_path_y, lw=0.5, fc="b", width=0.5)
plt.bar_label(buy_path_plt, label_type='edge')
plt.savefig("buy_path.jpg", bbox_inches='tight')