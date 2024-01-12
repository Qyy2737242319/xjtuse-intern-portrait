package cn.itcast.tags.meta

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 加载业务数据工具类：
 *      解析业务标签规则rule，依据规则判断数段数据源，加载业务数据
 */
object MetaParse extends Logging {
	
	/**
	 * 依据标签数据，获取业务标签规则rule，解析转换为Map集合
	 * @param tagDF 标签数据
	 * @return Map集合
	 */
	def parseRuleToParams(tagDF: DataFrame): Map[String, String] = {
		import tagDF.sparkSession.implicits._
		
		// 1. 4级标签规则rule
		val tagRule: String = tagDF
			.filter($"level" === 4)
			.head()
			.getAs[String]("rule")
		logInfo(s"==== 业务标签数据规则: {$tagRule} ====")
		
		// 2. 解析标签规则，先按照换行\n符分割，再按照等号=分割
		/*
			inType=hbase
			zkHosts=bigdata-cdh01.itcast.cn
			zkPort=2181
			hbaseTable=tbl_tag_logs
			family=detail
			selectFieldNames=global_user_id,loc_url,log_time
			whereCondition=log_time#day#30
		 */
		val paramsMap: Map[String, String] = tagRule
			.split("\n")
			.map{ line =>
				val Array(attrName, attrValue) = line.trim.split("=")
				(attrName, attrValue)
			}
			.toMap
		
		// 3. 返回集合Map
		paramsMap
	}
	
	/**
	 * 依据inType判断数据源，封装元数据Meta，加载业务数据
	 * @param spark SparkSession实例对象
	 * @param paramsMap 业务数据源参数集合
	 * @return
	 */
	def parseMetaToData(spark: SparkSession,
	                    paramsMap: Map[String, String]): DataFrame = {
		
		// 1. 从inType获取数据源
		val inType: String = paramsMap("inType")
		
		// 2. 判断数据源，封装Meta，获取业务数据
		val businessDF: DataFrame = inType.toLowerCase match {
			case "hbase" =>
				// 解析map集合，封装Meta实体类中
				val hbaseMeta = HBaseMeta.getHBaseMeta(paramsMap)
				// 加载业务数据
				spark.read
					.format("hbase")
					.option("zkHosts", hbaseMeta.zkHosts)
					.option("zkPort", hbaseMeta.zkPort)
					.option("hbaseTable", hbaseMeta.hbaseTable)
					.option("family", hbaseMeta.family)
					.option("selectFields", hbaseMeta.selectFieldNames)
					.option("filterConditions", hbaseMeta.filterConditions)
					.load()
			case "mysql" =>
    		    // 解析Map集合，封装MySQLMeta对象中
				val mysqlMeta = MySQLMeta.getMySQLMeta(paramsMap)
				// 从MySQL表加载业务数据
				spark.read
					.format("jdbc")
					.option("driver", mysqlMeta.driver)
					.option("url", mysqlMeta.url)
					.option("user", mysqlMeta.user)
					.option("password", mysqlMeta.password)
					.option("dbtable", mysqlMeta.sql)
					.load()
			case "hive" =>
				// Map集合，封装HiveMeta对象
				val hiveMeta: HiveMeta = HiveMeta.getHiveMeta(paramsMap)
				// 从Hive表加载数据, TODO：此时注意，如果标签模型业务数从Hive表加载，创建SparkSession对象时，集成Hive
				spark.read
    				.table(hiveMeta.hiveTable)
					// def select(cols: Column*): DataFrame,   selectFieldNames: _* -> 将数组转换可变参数传递
    				.select(hiveMeta.selectFieldNames: _*)
    				//.filter(hiveMeta.whereCondition)
			case "hdfs" =>
				// 解析Map集合，封装HdfsMeta对象中
				val hdfsMeta: HdfsMeta = HdfsMeta.getHdfsMeta(paramsMap)
				// 从HDFS加载CSV格式数据
				spark.read
					.option("sep", hdfsMeta.sperator)
					.option("header", "true")
					.option("inferSchema", "true")
    				.csv(hdfsMeta.inPath)
    				.select(hdfsMeta.selectFieldNames: _*)
			case "es" =>
				null
			case _ =>
				// 如果未获取到数据，直接抛出异常
				new RuntimeException("业务标签规则未提供数据源信息，获取不到业务数据，无法计算标签")
				null
		}
		
		// 3. 返回加载业务数据
		businessDF
	}
}
