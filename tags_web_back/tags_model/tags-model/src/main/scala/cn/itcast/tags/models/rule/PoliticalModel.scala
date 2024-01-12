package cn.itcast.tags.models.rule

import cn.itcast.tags.models.BasicModel
import cn.itcast.tags.tools.TagTools
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 标签模型开发：用户政治面貌标签
 */
class PoliticalModel extends BasicModel{
	/*
	328	政治面貌
		329	群众		    1
		330	党员		    2
		331	无党派人士	3
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session: SparkSession = businessDF.sparkSession
		import session.implicits._
		import org.apache.spark.sql.functions._
		
		/*
		root
		 |-- id: string (nullable = true)
		 |-- politicalface: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(10, truncate = false)
		
		/*
		root
		 |-- id: long (nullable = false)
		 |-- name: string (nullable = true)
		 |-- rule: string (nullable = true)
		 |-- level: integer (nullable = true)
		 */
		//tagDF.printSchema()
		//tagDF.show(10, truncate = false)
		
		// 调用工具类，计算标签：规则匹配类型标签
		val modelDF: DataFrame = TagTools.ruleMatchTag(
			businessDF, "politicalface", tagDF
		)
		//modelDF.printSchema()
		//modelDF.show(500, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
}


object PoliticalModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new PoliticalModel()
		tagModel.executeModel(328L)
	}
}
