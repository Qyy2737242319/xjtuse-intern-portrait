from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python spark SQL") \
    .getOrCreate()

# df = spark.createDataFrame([('tom', 20), ('jack', 40)], ['name', 'age'])
# df.select('name').show()
# rdd=spark.sparkContext.parallelize([('tom' , 2),('jerry' ,  1)])
# df=rdd.toDF(['name','age'])
# print(df.count())
rdd = spark.sparkContext.parallelize([('tom', 20), ('jack', 18)])
df = rdd.toDF(['name', 'age'])

df.printSchema()  # 打印schema
df.show(truncate=False)  # 打印数据