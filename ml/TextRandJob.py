from pyspark import HiveContext

from sparkexample.SparkSessionBase import SparkSessionBase


class TextRandJob(SparkSessionBase):
    SPARK_URL = "local"
    SPARK_APP_NAME = 'TextRandJob'
    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()

    def start(self):
        print(self.spark)
        hc=HiveContext(self.spark.sparkContext)
        hc.sql('show databases').show()
        hive_df=hc.table('default.my_group')
        print(hive_df.count())






if __name__ == '__main__':
    TextRandJob().start()