package cn.itcast.tags.models

import cn.itcast.tags.meta.HBaseMeta
import cn.itcast.tags.tools.HBaseTools
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
 * 标签基类，各个标签模型继承此类，实现其中打标签方法doTag即可
 */
trait BasicModel extends Logging{
	
	// 变量声明
	var spark: SparkSession = _
	
	// 1. 初始化：构建SparkSession实例对象
	def init(): Unit = {
		// a. 创建SparkConf,设置应用相关配置
		val sparkConf: SparkConf = new SparkConf()
			.setAppName(this.getClass.getSimpleName.stripSuffix("$"))
			.setMaster("local[4]")
			// 设置Shuffle分区数目
			.set("spark.sql.shuffle.partitions", "4")
			// 设置序列化为：Kryo
			.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
			.registerKryoClasses(
				Array(classOf[ImmutableBytesWritable], classOf[Result], classOf[Put])
			)
		// b. 建造者模式创建SparkSession会话实例对象
		spark = SparkSession.builder()
			.config(sparkConf)
			// 启用与Hive集成
			.enableHiveSupport()
			// 设置与Hive集成: 读取Hive元数据MetaStore服务
			.config("hive.metastore.uris", "thrift://bigdata-cdh01.itcast.cn:9083")
			// 设置数据仓库目录: 将SparkSQL数据库仓库目录与Hive数据仓库目录一致
			.config(
				"spark.sql.warehouse.dir", "hdfs://bigdata-cdh01.itcast.cn:8020/user/hive/warehouse"
			)
			.getOrCreate()
	}
	
	// 2. 准备标签数据：依据标签ID从MySQL数据库表tbl_basic_tag获取标签数据
	def getTagData(tagId: Long): DataFrame = {
		// 指定业务标签ID获取标签数据
		val tagTable: String =
			s"""
			  |(
			  |SELECT id, name, rule, level  FROM profile_tags.tbl_basic_tag WHERE id = $tagId
			  |UNION
			  |SELECT id, name, rule, level  FROM profile_tags.tbl_basic_tag WHERE pid = $tagId
			  |) AS tag_table
			  |""".stripMargin
		// 从MySQL数据库加载标签数据
		spark.read
			.format("jdbc")
			.option("driver", "com.mysql.jdbc.Driver")
			.option("url",
				"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC")
			.option("dbtable", tagTable)
			.option("user", "root")
			.option("password", "123456")
			.load()
	}
	
	// 3. 业务数据：依据业务标签规则rule，从数据源获取业务数据
	def getBusinessData(tagDF: DataFrame): DataFrame = {
		import tagDF.sparkSession.implicits._
		
		// a. 获取业务标签规则
		val tagRule: String = tagDF
			.filter($"level" === 4) // 业务标签属于4级标签
			.head() // 返回Row对象
			.getAs[String]("rule")
		//logWarning(s"==================< $tagRule >=====================")
		
		// b. 解析标签规则rule，封装值Map集合
		val tagRuleMap: Map[String, String] = tagRule
			// 按照换行符分割
			.split("\\n")
			// 再按照等号分割
			.map{line =>
				val Array(attrKey, attrValue) = line.trim.split("=")
				(attrKey, attrValue)
			}
			.toMap // 转换为Map集合
		logWarning(s"================= { ${tagRuleMap.mkString(", ")} } ================")
		
		// c. 判断数据源inType，读取业务数据
		var businessDF: DataFrame = null
		if("hbase".equals(tagRuleMap("inType").toLowerCase)){
			// 封装标签规则中数据源信息至HBaseMeta对象中
			val hbaseMeta: HBaseMeta = HBaseMeta.getHBaseMeta(tagRuleMap)
			// 从HBase表加载数据
			businessDF = HBaseTools.read(
				spark, hbaseMeta.zkHosts, hbaseMeta.zkPort, //
				hbaseMeta.hbaseTable, hbaseMeta.family, //
				hbaseMeta.selectFieldNames.split(",") //
			)
		}else{
			// 如果未获取到数据，直接抛出异常
			new RuntimeException("业务标签未提供数据源信息，获取不到业务数据，无法计算标签")
		}
		
		// d. 返回数据
		businessDF
	}
	
	// 4. 构建标签：依据业务数据和属性标签数据建立标签
	def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame
	
	// 5. 保存画像标签数据至HBase表
	def saveTag(modelDF: DataFrame): Unit = {
		HBaseTools.write(
			modelDF, "bigdata-cdh01.itcast.cn", "2181", //
			"tbl_profile", "user", "userId"
		)
	}
	
	// 6. 关闭资源：应用结束，关闭会话实例对象
	def close(): Unit = {
		// 应用结束，关闭资源
		if(null != spark) spark.stop()
	}
	
	// 规定标签模型执行流程顺序
	def executeModel(tagId: Long): Unit ={
		// a. 初始化
		init()
		try{
			// b. 获取标签数据
			val tagDF: DataFrame = getTagData(tagId)
			//basicTagDF.show()
			tagDF.persist(StorageLevel.MEMORY_AND_DISK)
			tagDF.count()
			
			// c. 获取业务数据
			val businessDF: DataFrame = getBusinessData(tagDF)
			//businessDF.show()
			
			// d. 计算标签
			val modelDF: DataFrame = doTag(businessDF, tagDF)
			//modelDF.show()
			
			// e. 保存标签
			if(null != modelDF) saveTag(modelDF)
			
			tagDF.unpersist()
		}catch {
			case e: Exception => e.printStackTrace()
		}finally {
			// f. 关闭资源
			close()
		}
	}
}
