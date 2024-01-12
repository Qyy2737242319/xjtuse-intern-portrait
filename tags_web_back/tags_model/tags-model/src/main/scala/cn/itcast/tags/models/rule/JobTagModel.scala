package cn.itcast.tags.models.rule

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.sql.DataFrame

class JobTagModel extends AbstractModel("职业标签", ModelType.MATCH){
	/*
	321	职业
		322	学生		1
		323	公务员	2
		324	军人		3
		325	警察		4
		326	教师		5
		327	白领		6
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		//businessDF.printSchema()
		//businessDF.show(10, truncate = false)
		
		//tagDF.printSchema()
		//tagDF.show(10, truncate = false)
		
		val modelDF: DataFrame = TagTools.ruleMatchTag(
			businessDF, "job", tagDF
		)
		modelDF.printSchema()
		modelDF.show(100, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
}

object JobTagModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new JobTagModel()
		tagModel.executeModel(321L, isHive = true)
	}
}