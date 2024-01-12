package cn.itcast.tags.models.rule

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
 * 规则匹配类型标签模型开发：用户政治面貌标签模型
 */
class PoliticalTagModel extends AbstractModel("政治面貌标签", ModelType.MATCH){
	/*
	328	政治面貌
		329	群众		    1
		330	党员		    2
		331	无党派人士	3
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		// 导入隐式转换
		import businessDF.sparkSession.implicits._
		
		/*
		root
		 |-- id: string (nullable = true)
		 |-- politicalface: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(20, truncate = false)
		
		/*
		root
		 |-- id: long (nullable = false)
		 |-- name: string (nullable = true)
		 |-- rule: string (nullable = true)
		 |-- level: integer (nullable = true)
		 */
		//tagDF.printSchema()
		//tagDF.filter($"level".equalTo(5)).show(10, truncate = false)
		
		val modelDF: DataFrame = TagTools.ruleMatchTag(
			businessDF, "politicalface", tagDF
		)
		//modelDF.show(100, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
}

object PoliticalTagModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new PoliticalTagModel()
		tagModel.executeModel(328L)
	}
}