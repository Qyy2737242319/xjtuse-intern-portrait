package cn.itcast.tags.models.rmd

import cn.itcast.tags.config.ModelConfig
import cn.itcast.tags.models.{AbstractModel, ModelType}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.storage.StorageLevel

import scala.util.matching.Regex

/**
 * 协同过滤推荐：用户购物偏好模型Bp
 */
class BpModel extends AbstractModel("用户购物偏好BP", ModelType.ML){
	/*
	382	用户购物偏好
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session = businessDF.sparkSession
		import session.implicits._
		
		/*
		root
			 |-- global_user_id: string (nullable = true)
			 |-- loc_url: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(30, truncate = false)
		
		/**
		 * TODO: 通过用户浏览行为日志数据，获取 用户对物品 点击浏览次数
		 *  1. 从loc_url中提取出 productId
		 *      自定义UDF函数
		 *  2. 过滤 productId不为null数据
		 *      userId    productId
		 *  3. 按照userId和productId分组，计算出 用户对物品 点击浏览次数
		 *      userId   productId   clickCount(隐式评价）
		 *  4. 使用ALS算法训练模型
		 *     考虑使用 交叉验证或者训练分割验证，调整超参数，获取最佳模型
		 *  5. 使用模型为用户推荐物品
		 *      Top5 物品
		 *  6. 将给用户推荐物品保存至HBase表中
		 *      rowkey: UserId， Column：products
		 */
		// 1. 自定义函数，从url中提取出访问商品id
		val url_to_product: UserDefinedFunction = udf(
			(url: String) => {
				// 正则表达式
				val regex: Regex = "^.+\\/product\\/(\\d+)\\.html.+$".r
				// 正则匹配
				val optionMatch: Option[Regex.Match] = regex.findFirstMatchIn(url)
				// 获取匹配的值
				val productId = optionMatch match {
					case Some(matchValue) => matchValue.group(1)
					case None => null
				}
				// 返回productId
				productId
			}
		)
		
		// 2. 从url中计算商品id
		val ratingsDF: Dataset[Row] = businessDF
			.filter($"loc_url".isNotNull) // 获取loc_url不为null
			.select(
				$"global_user_id".as("userId"), //
				url_to_product($"loc_url").as("productId") //
			)
			.filter($"productId".isNotNull) // 过滤不为空的数据
			// 统计每个用户点击各个商品的次数
			.groupBy($"userId", $"productId")
			.count()
			// 数据类型转换
			.select(
				$"userId".cast(DoubleType), //
				$"productId".cast(DoubleType), //
				$"count".as("rating").cast(DoubleType) //
			)
		/*
			root
			 |-- userId: double (nullable = true)
			 |-- productId: double (nullable = true)
			 |-- rating: double (nullable = false)
		 */
		//ratingsDF.printSchema()
		//ratingsDF.show(10, truncate = false)
		
		// 数据缓存
		ratingsDF.persist(StorageLevel.MEMORY_AND_DISK)
		
		// 3. 使用ALS算法训练模型（评分为隐式评分）
		val alsModel: ALSModel = new ALS()
			// 设置属性
			.setUserCol("userId")
			.setItemCol("productId")
			.setRatingCol("rating")
    		.setPredictionCol("prediction")
			// 设置算法参数
			.setImplicitPrefs(true) // 隐式评分
			.setRank(10) // 矩阵因子，rank秩的值
			.setMaxIter(10) // 最大迭代次数
			.setColdStartStrategy("drop") // 冷启动
			.setAlpha(1.0)
			.setRegParam(1.0)
			// 应用数据集，训练模型
			.fit(ratingsDF)
		ratingsDF.unpersist()
		
		// 4. 模型评估
		import org.apache.spark.ml.evaluation.RegressionEvaluator
		val evaluator: RegressionEvaluator = new RegressionEvaluator()
			.setLabelCol("rating")
			.setPredictionCol("prediction")
			.setMetricName("rmse")
		val rmse: Double = evaluator.evaluate(alsModel.transform(ratingsDF))
		//  rmse = 1.0300179222180903
		println(s"rmse = $rmse")
		
		// 5. 使用推荐模型推荐：
		/*
			推荐有两种方式：
			1）、方式一：给用户推荐商品
				当用户登录APP网站时，直接给用户推荐商品（Top10，Top5）
			2）、方式二：给商品推荐用户
				当用户登录APP网站时，将某个商品对给不同的用户
		 */
		// 5.1 给用户推荐商品: Top5
		val rmdItemsDF: DataFrame = alsModel.recommendForAllUsers(5)
		/*
		 |-- userId: integer (nullable = false)
		 |-- recommendations: array (nullable = true)
		 |    |-- element: struct (containsNull = true)
		 |    |    |-- productId: integer (nullable = true)
		 |    |    |-- rating: float (nullable = true)
		 */
		//rmdItemsDF.printSchema()
		//rmdItemsDF.show(10, truncate = false)
		
		// 5.2. 给物品推荐用户
		//val rmdUsersDF: DataFrame = alsModel.recommendForAllItems(5)
		/*
		root
		 |-- productId: integer (nullable = false)
		 |-- recommendations: array (nullable = true)
		 |    |-- element: struct (containsNull = true)
		 |    |    |-- userId: integer (nullable = true)
		 |    |    |-- rating: float (nullable = true)
		 */
		//rmdUsersDF.printSchema()
		//rmdUsersDF.show(10, truncate = false)
		
		/*
			TODO: 将推荐模型给用户推荐物品数据存储到HBase表中，依据用户Id直接获取推荐物品即可
				 create "tbl_rmd_items", "info"
		 */
		// 保存数据至HBase表的DataFrame数据集，要求其中所有列的值的类型必须是String类型
		// TODO: 将recommendations列array类型数据转换为字符串String类型
		val modelDF: DataFrame = rmdItemsDF
			.select(
				$"userId", //
				$"recommendations.productId".as("productIds"),
				$"recommendations.rating".as("ratings")
			)
			.select(
				$"userId".cast(StringType), //
				concat_ws(",", $"productIds").as("productIds"), //
				concat_ws(",", $"ratings").as("ratings") //
			)
		modelDF.printSchema()
		modelDF.show(10, truncate = false)
		modelDF.write
			.mode(SaveMode.Overwrite)
			.format("hbase")
			.option("zkHosts", ModelConfig.PROFILE_TABLE_ZK_HOSTS)
			.option("zkPort", ModelConfig.PROFILE_TABLE_ZK_PORT)
			.option("hbaseTable", "tbl_rmd_items")
			.option("family", "info")
			.option("rowKeyColumn", "userId")
			.save()
		
		// 最终标签返回值就是null
		null
	}
}

object BpModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new BpModel()
		tagModel.executeModel(382L)
	}
}