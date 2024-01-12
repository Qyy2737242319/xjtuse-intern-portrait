from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("RFMImport").getOrCreate()
# Specify the path to the CSV file
csv_file_path = "rfm.csv"

# Read the CSV file into a PySpark DataFrame
rfm_spark = spark.read.option("header", "true").csv(csv_file_path)
mysql_properties = {
    "driver": "com.mysql.cj.jdbc.Driver",
    "url": "jdbc:mysql://192.168.52.205:3306/tags_dat/rfm_table",
    "user": "root",
    "password": "admin",
    "dbtable": "rfm_table",  # Specify the target table name
}
# rfm_spark.write.jdbc(mysql_properties, mode="overwrite")
rfm_spark.write.jdbc(url=mysql_properties["url"], table=mysql_properties["dbtable"], mode="overwrite", properties=mysql_properties)
# mysql_properties = {
#     "url": "jdbc:mysql://19.168.52.205:3306/tags_dat?user=root&password=admin",
#     "dbtable": "rfm_table",  # 指定目标表名
# }
#
# # 将 PySpark DataFrame 写入 MySQL 数据库
# rfm_spark.write.jdbc(**mysql_properties, mode="overwrite")