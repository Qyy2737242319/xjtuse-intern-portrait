# from pyspark.sql.session import SparkSession as spark
#
# sc = spark.builder.master('local[*]').appName('pysparktest').getOrCreate()
#
# stuDF = sc.read.csv('UserBehavior.csv',header=True)
#
# stuDF.show()
# prop = {}
# prop['user'] = 'root'
# prop['password'] = 'admin'
# prop['driver'] = 'com.mysql.jdbc.Driver'
# stuDF.write.jdbc('jdbc:mysql://192.168.52.205:3306/tags_dat?characterEncoding=UTF-8','UserBehavior1','append',prop)
#
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

# Create a Spark session
spark = SparkSession.builder.master('local[*]').appName('pysparktest').getOrCreate()

# Define the schema with column names
schema = StructType([
    StructField("User_ID", StringType(), True),
    StructField("Item_ID", StringType(), True),
    StructField("CateGory_ID", StringType(), True),
    StructField("Behavior_type", StringType(), True),
    StructField("Timestamp", LongType(), True),
])

# Read CSV with the specified schema
stuDF = spark.read.csv('UserBehavior.csv', header=True, schema=schema)

# Show the DataFrame
stuDF.show()

# Define the database connection properties
prop = {}
prop['user'] = 'root'
prop['password'] = 'admin'
prop['driver'] = 'com.mysql.jdbc.Driver'

# Write DataFrame to MySQL database
stuDF.write.jdbc('jdbc:mysql://192.168.52.205:3306/tags_dat?characterEncoding=UTF-8', 'UserBehavior', 'append', prop)
