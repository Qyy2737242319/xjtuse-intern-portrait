package cn.itcast.tags.etl.hfile

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.hbase.client.{Connection, ConnectionFactory, Table}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.{HFileOutputFormat2, LoadIncrementalHFiles}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{HBaseConfiguration, KeyValue, TableName}
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.TreeMap

/**
 * 将数据存储文本文件转换为HFile文件，加载到HBase表中
 */
object HBaseBulkLoader {
	
	/**
	 * 依据不同表的数据文件，提取对应数据，封装到KeyValue对象中
	 */
	def getLineToData(line: String, family: String,
	                  fieldNames: TreeMap[String, Int]
	                 ): List[(ImmutableBytesWritable, KeyValue)] = {
		val length = fieldNames.size
		// 分割字符串
		val fieldValues: Array[String] = line.split("\\t", -1)
		if(null == fieldValues || fieldValues.length != length) return Nil
		
		// 获取id，构建RowKey
		val id: String = fieldValues(0)
		val rowKey = Bytes.toBytes(id)
		val ibw: ImmutableBytesWritable = new ImmutableBytesWritable(rowKey)
		
		// 列簇
		val columnFamily: Array[Byte] = Bytes.toBytes(family)
		
		// 构建KeyValue对象
		fieldNames.toList.map{ case (fieldName, fieldIndex) =>
			// KeyValue实例对象
			val keyValue = new KeyValue(
				rowKey, //
				columnFamily, //
				Bytes.toBytes(fieldName), //
				Bytes.toBytes(fieldValues(fieldIndex)) //
			)
			// 返回
			(ibw, keyValue)
		}
	}
	
	def main(args: Array[String]): Unit = {
		// 应用执行时传递5个参数：数据类型、HBase表名称、表列簇、输入路径及输出路径
		/*
		args = Array("1", "tbl_tag_logs", "detail", "/user/hive/warehouse/tags_dat.db/tbl_logs", "/datas/output_hfile/tbl_tag_logs")
		args = Array("2", "tbl_tag_users", "detail", "/user/hive/warehouse/tags_dat.db/tbl_users", "/datas/output_hfile/tbl_tag_users")
		args = Array("3", "tbl_tag_orders", "detail", "/user/hive/warehouse/tags_dat.db/tbl_orders", "/datas/output_hfile/tbl_tag_orders")
		args = Array("4", "tbl_tag_goods", "detail", "/user/hive/warehouse/tags_dat.db/tbl_goods", "/datas/output_hfile/tbl_tag_goods")
		*/
		// 将传递赋值给变量， 其中数据类型：1Log、2User、3Order、4Good
		val Array(dataType, tableName, family, inputDir, outputDir) = Array(
			"3", "tbl_tag_orders", "detail",
			"/user/hive/warehouse/tags_dat.db/tbl_orders", "/datas/output_hfile/tbl_tag_orders"
		)
		
		// 依据参数获取处理数据schema
		val fieldNames: TreeMap[String, Int] = dataType.toInt match {
			case 1 => TableFieldNames.LOG_FIELD_NAMES
			case 2 => TableFieldNames.USER_FIELD_NAMES
			case 3 => TableFieldNames.ORDER_FIELD_NAMES
			case 4 => TableFieldNames.GOODS_FIELD_NAMES
		}
		
		// 1. 构建SparkContext实例对象
		val sc: SparkContext = {
			// a. 创建SparkConf，设置应用配置信息
			val sparkConf = new SparkConf()
				.setMaster("local[2]")
				.setAppName(this.getClass.getSimpleName.stripSuffix("$"))
				.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
			// b. 传递SparkContext创建对象
			SparkContext.getOrCreate(sparkConf)
		}
		
		// 2. 读取文本文件数据，转换格式
		val keyValuesRDD: RDD[(ImmutableBytesWritable, KeyValue)] = sc
			.textFile(inputDir)
			// 过滤数据
			.filter(line => null != line)
			// 提取数据字段，构建二元组(RowKey, KeyValue)
			/*
				Key: rowkey + cf + column + version(timestamp)
				Value: ColumnValue
			 */
			.flatMap{line => getLineToData(line, family, fieldNames)}
			// TODO: 对数据做字典排序
			.sortByKey()
		
		// TODO：构建Job，设置相关配置信息，主要为输出格式
		// a. 读取配置信息
		val conf: Configuration = HBaseConfiguration.create()
		// b. 如果输出目录存在，删除
		val dfs: FileSystem = FileSystem.get(conf)
		val outputPath: Path = new Path(outputDir)
		if(dfs.exists(outputPath)){
			dfs.delete(outputPath, true)
		}
		dfs.close()
		
		// TODO：c. 配置HFileOutputFormat2输出
		val conn: Connection = ConnectionFactory.createConnection(conf)
		val htableName = TableName.valueOf(tableName)
		val table: Table = conn.getTable(htableName)
		HFileOutputFormat2.configureIncrementalLoad(
			Job.getInstance(conf), //
			table, //
			conn.getRegionLocator(htableName)//
		)
		
		// TODO： 3. 保存数据为HFile文件
		keyValuesRDD.saveAsNewAPIHadoopFile(
			outputDir, //
			classOf[ImmutableBytesWritable], //
			classOf[KeyValue], //
			classOf[HFileOutputFormat2], //
			conf //
		)
		
		// TODO：4. 将输出HFile加载到HBase表中
		val load = new LoadIncrementalHFiles(conf)
		load.doBulkLoad(
			outputPath, conn.getAdmin, table, conn.getRegionLocator(htableName)
		)
		
		// 应用结束，关闭资源
		sc.stop()
	}
}