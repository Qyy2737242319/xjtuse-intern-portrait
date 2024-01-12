package cn.itcast.tags.models.rule

import cn.itcast.tags.meta.HBaseMeta
import cn.itcast.tags.tools.HBaseTools
import org.apache.hadoop.hbase.client.{Put, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 标签模型开发：用户性别标签模型
 */
object GenderModel extends Logging{
	
	/*
	318	性别
		319	男		1
		320	女		2
	 */
	def main(args: Array[String]): Unit = {
		
		// TODO: 1. 创建SparkSession实例对象
		val spark: SparkSession = {
			// 1.a. 创建SparkConf 设置应用信息
			val sparkConf = new SparkConf()
				.setAppName(this.getClass.getSimpleName.stripSuffix("$"))
				.setMaster("local[4]")
				.set("spark.sql.shuffle.partitions", "4")
				// 由于从HBase表读写数据，设置序列化
				.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
				.registerKryoClasses(
					Array(classOf[ImmutableBytesWritable], classOf[Result], classOf[Put])
				)
			// 1.b. 建造者模式构建SparkSession对象
			val session = SparkSession.builder()
				.config(sparkConf)
				.getOrCreate()
			// 1.c. 返回会话实例对象
			session
		}
		import spark.implicits._
		import org.apache.spark.sql.functions._
		
		// 2. 依据TagId，从MySQL读取标签数据(4级业务标签和5级属性标签）
		val tagTable: String =
			"""
			  |(
			  |SELECT id, name, rule, level  FROM profile_tags.tbl_basic_tag WHERE id = 318
			  |UNION
			  |SELECT id, name, rule, level  FROM profile_tags.tbl_basic_tag WHERE pid = 318
			  |) AS tag_table
			  |""".stripMargin
		val basicTagDF: DataFrame = spark.read
			.format("jdbc")
			.option("driver", "com.mysql.jdbc.Driver")
			.option("url",
				"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC")
			.option("dbtable", tagTable)
			.option("user", "root")
			.option("password", "123456")
			.load()
		/*
		root
		 |-- id: long (nullable = false)
		 |-- name: string (nullable = true)
		 |-- rule: string (nullable = true)
		 |-- level: integer (nullable = true)
		 */
		//basicTagDF.printSchema()
		//basicTagDF.show(10, truncate = false)
		
		// 3. 解析标签rule，从HBase读取业务数据
		// 3.1 获取业务标签规则
		val tagRule: String = basicTagDF
			.filter($"level" === 4) // 业务标签属于4级标签
			.head() // 返回Row对象
			.getAs[String]("rule")
		//logWarning(s"==================< $tagRule >=====================")
		
		// 3.2 解析标签规则rule，封装值Map集合
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
		
		// 3.3 判断数据源inType，读取业务数据
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
		/*
		root
		 |-- id: string (nullable = true)
		 |-- gender: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(100, truncate = false)
		
		// 4. 业务数据结合标属性签数据，构建标签
		// 4.1 获取属性标签规则
		val attrTagRuleDF: DataFrame = basicTagDF
			.filter($"level" === 5) // 属性标签为5级标签
			.select($"rule", $"name")
		// 4.2 将业务数据和属性标签数据，按照字段进行关联JOIN，属于等值JOIN
		val modelDF: DataFrame = businessDF
			.join(
				attrTagRuleDF, //
				businessDF("gender") === attrTagRuleDF("rule"), //
				"inner" //
			)
			.select(
				$"id".as("userId"), //
				$"name".as("gender") //
			)
		/*
		root
		 |-- userId: string (nullable = true)
		 |-- gender: string (nullable = true)
		 */
		//modelDF.printSchema()
		//modelDF.show(500, truncate = false)
		
		// 5. 画像标签数据存储HBase表
		HBaseTools.write(
			modelDF, "bigdata-cdh01.itcast.cn", "2181", //
			"tbl_profile", "user", "userId"
		)
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
