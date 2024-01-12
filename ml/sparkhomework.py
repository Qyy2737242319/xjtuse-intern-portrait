from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, concat, countDistinct, to_timestamp, year, count, month, to_date
from pyspark.sql.types import StructType, ArrayType, StringType, StructField, IntegerType, BooleanType, FloatType

# Driver
spark = SparkSession \
    .builder \
    .master('local') \
    .appName('HelloSpark') \
    .getOrCreate()

fire_schema = StructType([StructField("CallNumber", IntegerType(), True),
                          StructField("UnitID", StringType(), True),
                          StructField("IncidentNumber", IntegerType(), True),
                          StructField("CallType", StringType(), True),
                          StructField("CallDate", StringType(), True),
                          StructField("WatchDate", StringType(), True),
                          StructField("CallFinalDisposition", StringType(), True),
                          StructField("AvailableDtTm", StringType(), True),
                          StructField("Address", StringType(), True),
                          StructField("City", StringType(), True),
                          StructField("Zipcode", IntegerType(), True),
                          StructField("Battalion", StringType(), True),
                          StructField("StationArea", StringType(), True),
                          StructField("Box", StringType(), True),
                          StructField("OriginalPriority", StringType(), True),
                          StructField("Priority", StringType(), True),
                          StructField("FinalPriority", IntegerType(), True),
                          StructField("ALSUnit", BooleanType(), True),
                          StructField("CallTypeGroup", StringType(), True),
                          StructField("NumAlarms", IntegerType(), True),
                          StructField("UnitType", StringType(), True),
                          StructField("UnitSequenceInCallDispatch", IntegerType(), True),
                          StructField("FirePreventionDistrict", StringType(), True),
                          StructField("SupervisorDistrict", StringType(), True),
                          StructField("Neighborhood", StringType(), True),
                          StructField("Location", StringType(), True),
                          StructField("RowID", StringType(), True),
                          StructField("Delay", FloatType(), True)
                          ]
                         )

df = spark.read.option('header', True).schema(fire_schema).csv('dataset/sf-fire-calls.txt')
print("--------------------------------------------------------------------------------")
df.select('IncidentNumber', 'AvailableDtTm', 'CallType') \
        .where("CallType == 'Medical Incident'") \
        .show(truncate=False)
print("--------------------------------------------------------------------------------")
df.select('CallType') \
    .where(col('CallType').isNotNull()) \
    .agg(countDistinct('CallType')) \
    .show(truncate=False)
print("--------------------------------------------------------------------------------")
df.select('CallType') \
    .where(col('CallType').isNotNull()) \
    .distinct().show(truncate=False)
print("--------------------------------------------------------------------------------")
df.withColumnRenamed('Delay', 'ResponseDelayedinMins') \
    .where('ResponseDelayedinMins > 5') \
    .select('ResponseDelayedinMins').show(truncate=False)
print("--------------------------------------------------------------------------------")
cleaned_df = df.withColumn('IncidentDate', to_timestamp(col('CallDate'), 'MM/dd/yyyy')) \
    .drop('CallDate') \
    .withColumn("OnWatchDate", to_timestamp(col("WatchDate"), "MM/dd/yyyy")) \
    .drop("WatchDate") \
    .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")) \
    .drop("AvailableDtTm")

cleaned_df.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, True)
print("--------------------------------------------------------------------------------")
cleaned_df.select(year('IncidentDate').alias('year')) \
    .distinct() \
    .orderBy(col('year').asc()) \
    .show(truncate=False)
print("--------------------------------------------------------------------------------")
df.where(col('CallType').isNotNull())\
    .groupby('CallType') \
    .agg(count('CallType').alias('count')) \
    .orderBy(col('count').desc()).show(truncate=False)
print("打印2018年份所有的CallType，并去重")
df.select('CallType') \
    .where(col('CallType').isNotNull()&col('CallDate').like('%/2018')) \
    .distinct().show(truncate=False)
print("2018年的哪个月份有最高的火警")
homework2_df=cleaned_df.select(month('IncidentDate').alias('month'))\
              .where(year('IncidentDate')==2018)\
              .groupBy('month')\
              .agg(count('month').alias('count'))\
              .orderBy(col('count').desc())
homework2_df.show(1,truncate=False)
file=r"iris.parquet"
df=spark.read.parquet(file)
print("第七问：")
df.show()
string_indexer = StringIndexer(inputCol='CallNumber', outputCol='CallNumber1')

# 在数据上拟合 StringIndexer 模型
model = string_indexer.fit(df)

# 使用模型进行转换
df_indexed = model.transform(df)

# 打印转换后的 DataFrame
df_indexed.select('CallNumber', 'CallNumber1').show()

spark.read.format("jdbc")\
  .option("url", "jdbc:mysql://192.168.52.206:3306/mysql")\
  .option("dbtable", "user")\
  .option("user", "root")\
  .option("password", "admin")\
  .load()\
  .show()