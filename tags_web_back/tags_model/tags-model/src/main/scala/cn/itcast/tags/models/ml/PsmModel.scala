package cn.itcast.tags.models.ml

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.{MLModelTools, TagTools}
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DataTypes

/**
 * 挖掘类型标签模型开发：价格敏感度标签模型
 */
class PsmModel extends AbstractModel("价格敏感度PSM", ModelType.ML){
	/*
	372	消费敏感度
		373	极度敏感		0
		374	比较敏感		1
		375	一般敏感		2
		376	不太敏感		3
		377	极度不敏感	4
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session = businessDF.sparkSession
		import session.implicits._
		
		/*
			root
			 |-- memberid: string (nullable = true)
			 |-- ordersn: string (nullable = true)
			 |-- orderamount: string (nullable = true)
			 |-- couponcodevalue: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(30, truncate = false)
		
		//tagDF.printSchema()
		/*
			+---+-----+----+-----+
			|id |name |rule|level|
			+---+-----+----+-----+
			|373|极度敏感 |0   |5    |
			|374|比较敏感 |1   |5    |
			|375|一般敏感 |2   |5    |
			|376|不太敏感 |3   |5    |
			|377|极度不敏感|4   |5    |
			+---+-----+----+-----+
		 */
		//tagDF.filter($"level" === 5).show(10, truncate = false)
		
		/*
		TODO: 1、依据业务数据（订单数据）计算每个用户PSM值
				psm = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
					 tdonr			adar		tdar
				tdonr 优惠订单占比(优惠订单数 / 订单总数)
				adar  平均优惠金额占比(平均优惠金额 / 平均每单应收金额)
					平均优惠金额 = 优惠总金额 / 优惠订单数
					如果某个人不差钱，购物从来不使用优惠，此时优惠订单数为0，分母为0，计算出 平均优惠金额为null
				tdar  优惠总金额占比(优惠总金额 / 订单总金额)
		 */
		// 计算指标
		//ra: receivableAmount 应收金额
		val raColumn: Column = ($"orderamount" + $"couponcodevalue").as("ra")
		//da: discountAmount 优惠金额
		val daColumn: Column = $"couponcodevalue".cast(DataTypes.createDecimalType(10, 2)).as("da")
		//pa: practicalAmount 实收金额
		val paColumn: Column = $"orderamount".cast(DataTypes.createDecimalType(10, 2)).as("pa")
		//state: 订单状态，此订单是否是优惠订单，0表示非优惠订单，1表示优惠订单
		val stateColumn: Column = when($"couponcodevalue" === 0.0, 0)
			.otherwise(1).as("state")
		
		//tdon 优惠订单数
		val tdonColumn: Column = sum($"state").as("tdon")
		//ton  总订单总数
		val tonColumn: Column = count($"state").as("ton")
		//tda 优惠总金额
		val tdaColumn: Column  = sum($"da").as("tda")
		//tra 应收总金额
		val traColumn: Column = sum($"ra").as( "tra")
		
		/*
			tdonr 优惠订单占比(优惠订单数 / 订单总数)
			tdar  优惠总金额占比(优惠总金额 / 订单总金额)
			adar  平均优惠金额占比(平均优惠金额 / 平均每单应收金额)
		*/
		val tdonrColumn: Column = ($"tdon" / $"ton").as("tdonr")
		val tdarColumn: Column = ($"tda" / $"tra").as("tdar")
		val adarColumn: Column = (
			($"tda" / $"tdon") / ($"tra" / $"ton")
			).as("adar")
		val psmColumn: Column = ($"tdonr" + $"tdar" + $"adar").as("psm")
		
		val psmDF: DataFrame = businessDF
			// 确定每个订单ra、da、pa及是否为优惠订单
			.select(
				$"memberid".as("userId"), //
				raColumn, daColumn, paColumn, stateColumn //
			)
			// 按照userId分组，聚合统计：订单总数和订单总额
			.groupBy($"userId")
			.agg(
				tonColumn, tdonColumn, traColumn, tdaColumn
			)
			// 计算优惠订单占比、优惠总金额占比、adar
			.select(
				$"userId", tdonrColumn, tdarColumn, adarColumn
			)
			// 计算PSM值
			.select($"*", psmColumn)
			.select(
				$"*", //
				when($"psm".isNull, 0.00000001)
					.otherwise($"psm").as("psm_score")
			)
		/*
		root
		 |-- userId: string (nullable = true)
		 |-- tdonr: double (nullable = true)
		 |-- tdar: double (nullable = true)
		 |-- adar: double (nullable = true)
		 |-- psm: double (nullable = true)
		 |-- psm_score: double (nullable = true)
		 */
		//psmDF.printSchema()
		//psmDF.show(50, truncate = false)
		
		/*
		TODO: 2、使用KMeans聚类算法训练模型
			针对psm单列值聚类操作
		 */
		// 2.1. 特征值features
		val psmFeaturesDF: DataFrame = new VectorAssembler()
			.setInputCols(Array("psm_score"))
			.setOutputCol("features")
			.transform(psmDF)
		
		// 2.2. 获取KMeans模型
		val kmeansModel: KMeansModel = MLModelTools
			.loadModel(psmFeaturesDF, "psm", this.getClass).asInstanceOf[KMeansModel]
		
		/*
		TODO: 3、使用模型和属性标签规则打标签
		 */
		// 3.1. 模型预测
		val predictionDF: DataFrame = kmeansModel.transform(psmFeaturesDF)
		//predictionDF.printSchema()
		predictionDF.show(50, truncate = false)
		
		// 5. 打标签
		val modelDF: DataFrame = TagTools.kmeansMatchTag(kmeansModel, predictionDF, tagDF, "psm")
		modelDF.show(100, truncate = false)
		
		//  返回标签数据
		modelDF
	}
}

object PsmModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new PsmModel()
		tagModel.executeModel(372L)
	}
}