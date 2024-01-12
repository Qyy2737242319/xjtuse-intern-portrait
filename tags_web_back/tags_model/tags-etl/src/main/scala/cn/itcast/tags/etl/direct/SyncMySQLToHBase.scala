package cn.itcast.tags.etl.direct

import java.util.Properties

import org.apache.commons.lang3.StringUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Column, DataFrame, SparkSession}

object SyncMySQLToHBase {
	
	/**
	 * 将MySQL表中的数据同步到HBase表中，SparkSQL直接从MySQL表读取数据，保存到HBase表中
	 *
	 * @param spark SparkSession 会话上下文实例对象
	 * @param dbTable 数据库中表名称
	 * @param hbaseTable HBase表名称
	 * @param family HBase表列簇
	 * @param rowKeyColumn HBase表RowKey为MySQL表列名称
	 */
	def syncTable(spark: SparkSession, dbTable: String, hbaseTable: String,
	              family: String, rowKeyColumn: String): Unit = {
		import spark.implicits._
		import org.apache.spark.sql.functions._
		// 1. 直接从MySQL数据库表读取数据
		val tableDF: DataFrame = spark.read
			.jdbc(
				"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/tags_dat?user=root&password=123456&driver=com.mysql.jdbc.Driver", //
				dbTable,
				new Properties() //
			)
		val columnsString: Array[Column] = tableDF.columns.map{ columnName =>
			col(columnName).cast(StringType)
		}
		val datasDF: DataFrame = tableDF.select(columnsString: _*)
		
		// 2. 获取列名称
		val fieldNames: Array[String] = datasDF.columns
		
		// 3. DataFrame转换为RDD，对每个分区数据进行转换为(ImmutableBytesWritable, Put)
		val cfBytes: Array[Byte] = Bytes.toBytes(family)
		val putsRDD: RDD[(ImmutableBytesWritable, Put)] = datasDF.rdd.map{row =>
			// 获取RowKey对应的Column的值
			val rowKeyValue: String = row.getAs[String](rowKeyColumn)
			// 构建Put对象
			val put = new Put(Bytes.toBytes(rowKeyValue))
			// 遍历列名，获取列值，加入Put中
			fieldNames.foreach{fieldName =>
				val columnValue = row.getAs[String](fieldName)
				if(StringUtils.isNotBlank(columnValue)){
					put.addColumn( //
						cfBytes, //
						Bytes.toBytes(fieldName.toLowerCase()), //
						Bytes.toBytes(columnValue) //
					)
				}
			}
			// 返回二元组
			(new ImmutableBytesWritable(put.getRow), put)
		}
		
		// 4. 保存到HBase表中
		/*
		  def saveAsNewAPIHadoopFile(
		      path: String,
		      keyClass: Class[_],
		      valueClass: Class[_],
		      outputFormatClass: Class[_ <: NewOutputFormat[_, _]],
		      conf: Configuration = self.context.hadoopConfiguration
		  ): Unit
		 */
		// 读取配置信息
		val hbaseConf: Configuration = new Configuration()
		hbaseConf.set("hbase.zookeeper.quorum", "bigdata-cdh01.itcast.cn")
		hbaseConf.set("hbase.zookeeper.property.clientPort", "2181")
		// 设置表的名称
		hbaseConf.set(TableOutputFormat.OUTPUT_TABLE, hbaseTable)
		putsRDD.saveAsNewAPIHadoopFile(
			s"/datas/hbase/$dbTable-${System.currentTimeMillis()}", //
			classOf[ImmutableBytesWritable], //
			classOf[Put], //
			classOf[TableOutputFormat[ImmutableBytesWritable]], //
			hbaseConf //
		)
	}
	
	def main(args: Array[String]): Unit = {
		
		val dbTables: Array[String] = Array("tbl_users", "tbl_orders", "tbl_goods", "tbl_logs")
		val hbaseTables: Array[String] = Array("tbl_tag_users", "tbl_tag_orders", "tbl_tag_goods", "tbl_tag_logs")
		val family: String = "detail"
		
		// 1. 构建SparkSession实例对象
		val spark: SparkSession = SparkSession.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[2]")
			.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
			.config("spark.sql.shuffle.partitions", 4)
			.getOrCreate()
		
		// 同步MySQL的tbl_users表数据到HBase的tbl_users表中
		//syncTable(spark, dbTables(0), hbaseTables(0), family, "id");
		syncTable(spark, "tbl_profile_users", "tbl_profile_users", family, "id");
		
		// 同步MySQL的tbl_orders表数据到HBase的tbl_orders表中
		//syncTable(spark, dbTables(1), hbaseTables(1), family, "id");
		//syncTable(spark, "tbl_tag_orders", "tbl_tag_orders", family, "id");
		
		// 同步MySQL的tbl_goods表数据到HBase的tbl_goods表中
		//syncTable(spark, dbTables(2), hbaseTables(2), family, "id");
		//syncTable(spark, "tbl_goods", "tbl_tag_goods", family, "id");
		
		// 同步MySQL的tbl_logs表数据到HBase的tbl_logs表中
		//syncTable(spark, dbTables(3), hbaseTables(3), family, "id");
		//syncTable(spark, "tbl_logs", "tbl_tag_logs", family, "id");
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
