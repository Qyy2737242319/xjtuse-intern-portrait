package cn.itcast.tags.models.rule

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.sql.DataFrame

/**
 * 标签模型开发：用户国籍标签模型
 */
class NationalityTagModel extends AbstractModel("国籍标签", ModelType.MATCH){
	/*
	332	国籍
		333	中国大陆		1
		334	中国香港		2
		335	中国澳门		3
		336	中国台湾		4
		337	其他		    5
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		// TODO: 将业务数据中业务字段field与属性标签中规则rule进行匹配关联，给用户打上标签的值（TagName）
		import businessDF.sparkSession.implicits._
		
		/*
		root
		 |-- id: string (nullable = true)
		 |-- nationality: string (nullable = true)
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
		//tagDF.filter($"level" === 5).show(10, truncate = false)
		
		// 业务字段：nationality，与属性标签规则数据打标签，采用UDF函数
		val modelDF: DataFrame = TagTools.ruleMatchTag(
			businessDF, "nationality", tagDF
		)
		modelDF.show(10, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
}

object NationalityTagModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new NationalityTagModel()
		tagModel.executeModel(332L)
	}
}