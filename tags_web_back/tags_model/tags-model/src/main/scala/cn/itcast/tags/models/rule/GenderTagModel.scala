package cn.itcast.tags.models.rule

import cn.itcast.tags.models.{AbstractTagModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.sql.DataFrame

/**
 * 标签模型开发：用户性别标签模型
 */
class GenderTagModel extends AbstractTagModel("性别标签", ModelType.MATCH){
	/*
	318	性别
		319	男		1
		320	女		2
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val modelDF: DataFrame = TagTools.ruleMatchTag(
			businessDF, "gender", tagDF
		)
		// 返回画像标签数据
		modelDF
	}
}

object GenderTagModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new GenderTagModel()
		tagModel.executeModel(318L)
	}
}