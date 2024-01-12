package cn.itcast.tags.models

import cn.itcast.tags.config.ModelConfig
import cn.itcast.tags.meta.{HBaseMeta, MetaParse}
import cn.itcast.tags.utils.SparkUtils
import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
 * 标签基类，各个标签模型继承此类，实现其中打标签方法doTag即可
 */
abstract class AbstractTagModel(modelName: String, modelType: ModelType) extends Logging{
	
	// 设置Spark应用程序运行的用户：root, 默认情况下为当前系统用户
	System.setProperty("user.name", ModelConfig.FS_USER)
	System.setProperty("HADOOP_USER_NAME", ModelConfig.FS_USER)
	
	// 变量声明
	var spark: SparkSession = _
	
	// 1. 初始化：构建SparkSession实例对象
	def init(isHive: Boolean = false): Unit = {
		spark = SparkUtils.createSparkSession(this.getClass, isHive)
	}
	
	// 2. 准备标签数据：依据标签ID从MySQL数据库表tbl_basic_tag获取标签数据
	def getTagData(tagId: Long): DataFrame = {
		spark.read
			.format("jdbc")
			.option("driver", ModelConfig.MYSQL_JDBC_DRIVER)
			.option("url", ModelConfig.MYSQL_JDBC_URL)
			.option("dbtable", ModelConfig.tagTable(tagId))
			.option("user", ModelConfig.MYSQL_JDBC_USERNAME)
			.option("password", ModelConfig.MYSQL_JDBC_PASSWORD)
			.load()
	}
	
	// 3. 业务数据：依据业务标签规则rule，从数据源获取业务数据
	def getBusinessData(tagDF: DataFrame): DataFrame = {
		// a. 获取业务标签规则rule，并解析封装值Map集合
		val rulesMap = MetaParse.parseRuleToParams(tagDF)
		// b. 依据inType判断数据源，加载业务数据
		val businessDF: DataFrame = MetaParse.parseMetaToData(spark, rulesMap)
		// c. 返回加载业务数据
		businessDF
	}
	
	// 4. 构建标签：依据业务数据和属性标签数据建立标签
	def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame
	
	// 5. 保存画像标签数据至HBase表
	def saveTag(modelDF: DataFrame): Unit = {
		/*
			HBaseTools.write(
				modelDF, //
				ModelConfig.PROFILE_TABLE_ZK_HOSTS, //
				ModelConfig.PROFILE_TABLE_ZK_PORT, //
				ModelConfig.PROFILE_TABLE_NAME, //
				ModelConfig.PROFILE_TABLE_FAMILY_USER, //
				ModelConfig.PROFILE_TABLE_ROWKEY_COL //
			)
		 */
		modelDF.write
			.mode(SaveMode.Overwrite)
			.format("hbase")
			.option("zkHosts", ModelConfig.PROFILE_TABLE_ZK_HOSTS)
			.option("zkPort", ModelConfig.PROFILE_TABLE_ZK_PORT)
			.option("hbaseTable", ModelConfig.PROFILE_TABLE_NAME)
			.option("family", ModelConfig.PROFILE_TABLE_FAMILY_USER)
			.option("rowKeyColumn", ModelConfig.PROFILE_TABLE_ROWKEY_COL)
			.save()
	}
	
	// 6. 关闭资源：应用结束，关闭会话实例对象
	def close(): Unit = {
		// 应用结束，关闭资源
		if(null != spark) spark.stop()
	}
	
	// 规定标签模型执行流程顺序
	def executeModel(tagId: Long, isHive: Boolean = false): Unit ={
		// a. 初始化
		init(isHive)
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
