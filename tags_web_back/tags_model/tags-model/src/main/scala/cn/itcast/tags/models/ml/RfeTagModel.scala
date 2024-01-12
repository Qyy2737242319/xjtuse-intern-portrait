package cn.itcast.tags.models.ml

import cn.itcast.tags.config.ModelConfig
import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.{MLModelTools, TagTools}
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 挖掘类型标签模型开发：用户活跃度标签模型
 */
class RfeTagModel extends AbstractModel("用户活跃度RFE", ModelType.ML){
	/*
	367	用户活跃度
		368	非常活跃		0
		369	活跃		    1
		370	不活跃		2
		371	非常不活跃	3
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session: SparkSession = businessDF.sparkSession
		import session.implicits._
		
		/*
		root
		 |-- global_user_id: string (nullable = true)
		 |-- loc_url: string (nullable = true)
		 |-- log_time: string (nullable = true)
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
		
		// 1. 从业务数据中计算R、F、E值
		val rfeDF: DataFrame = businessDF
			.groupBy($"global_user_id") // 按照用户分组
			// 使用函数计算R、F、E值
			.agg(
				max($"log_time").as("last_time"), //
				// 计算F值
				count($"loc_url").as("frequency"), //
				// 计算E值
				countDistinct($"loc_url").as("engagements")
			)
			.select(
				$"global_user_id".as("userId"), //
				// 计算R值
				datediff(
					date_sub(current_timestamp(), 200), $"last_time"//
				).as("recency"), //
				$"frequency", $"engagements" //
			)
		/*
			root
			 |-- userId: string (nullable = true)
			 |-- recency: integer (nullable = true)
			 |-- frequency: long (nullable = false)
			 |-- engagements: long (nullable = false)
		 */
		//rfeDF.printSchema()
		/*
			+------+-------+---------+-----------+
			|userId|recency|frequency|engagements|
			+------+-------+---------+-----------+
			|1     |23     |418      |270        |
			|102   |23     |415      |271        |
			|107   |23     |424      |280        |
			|110   |23     |356      |249        |
			|111   |23     |381      |257        |
			|120   |23     |386      |269        |
		 */
		//rfeDF.show(50, truncate = false)
		
		// 2. 按照规则，给RFE值打分: Score
		/*
			R:0-15天=5分，16-30天=4分，31-45天=3分，46-60天=2分，大于61天=1分
			F:≥400=5分，300-399=4分，200-299=3分，100-199=2分，≤99=1分
			E:≥250=5分，200-249=4分，150-199=3分，149-50=2分，≤49=1分
		 */
		// R 打分条件表达式
		val rWhen = when(col("recency").between(1, 15), 5.0) //
			.when(col("recency").between(16, 30), 4.0) //
			.when(col("recency").between(31, 45), 3.0) //
			.when(col("recency").between(46, 60), 2.0) //
			.when(col("recency").geq(61), 1.0) //
		// F 打分条件表达式
		val fWhen = when(col("frequency").leq(99), 1.0) //
			.when(col("frequency").between(100, 199), 2.0) //
			.when(col("frequency").between(200, 299), 3.0) //
			.when(col("frequency").between(300, 399), 4.0) //
			.when(col("frequency").geq(400), 5.0) //
		// M 打分条件表达式
		val eWhen = when(col("engagements").lt(49), 1.0) //
			.when(col("engagements").between(50, 149), 2.0) //
			.when(col("engagements").between(150, 199), 3.0) //
			.when(col("engagements").between(200, 249), 4.0) //
			.when(col("engagements").geq(250), 5.0) //
		val rfeScoreDF: DataFrame = rfeDF.select(
			$"userId",
			rWhen.as("r_score"), //
			fWhen.as("f_score"), //
			eWhen.as("e_score") //
		)
		//rfeScoreDF.printSchema()
		//rfeScoreDF.show()
		
		// TODO： 3. 使用计算rfe值，应用到聚类KMeans算法中，训练模型，获取最佳模型，且预测
		// 3.1 提取特征，封装features向量中
		val assembler: VectorAssembler = new VectorAssembler()
			.setInputCols(Array("r_score", "f_score", "e_score"))
			.setOutputCol("features")
		val rfeFeaturesDF: DataFrame = assembler.transform(rfeScoreDF)
		
		// 3.2 获取最佳模型
		val kmeansModel: KMeansModel = MLModelTools
			.loadModel(rfeFeaturesDF, "rfe", this.getClass).asInstanceOf[KMeansModel]
		
		// TODO: 4. 打标签
		// 4.1 使用模型预测
		val predictionDF: DataFrame = kmeansModel.transform(rfeFeaturesDF)
		/*
			+------+-------+-------+-------+-------------+----------+
			|userId|r_score|f_score|e_score|features     |prediction|
			+------+-------+-------+-------+-------------+----------+
			|1     |1.0    |5.0    |5.0    |[1.0,5.0,5.0]|0         |
			|110   |1.0    |4.0    |5.0    |[1.0,4.0,5.0]|1         |
			|130   |1.0    |5.0    |4.0    |[1.0,5.0,4.0]|3         |
			|135   |1.0    |4.0    |4.0    |[1.0,4.0,4.0]|2         |
		 */
		//predictionDF.show(20, truncate = false)
		
		// 4.2 打标签
		val modelDF: DataFrame = TagTools.kmeansMatchTag(
			kmeansModel, predictionDF, tagDF, "rfe"
		)
		//modelDF.printSchema()
		modelDF.show(100, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
}

object RfeTagModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new RfeTagModel()
    	tagModel.executeModel(367L)
	}
}