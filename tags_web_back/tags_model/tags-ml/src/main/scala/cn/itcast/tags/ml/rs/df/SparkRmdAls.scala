package cn.itcast.tags.ml.rs.df

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}

/**
 * 基于DataFrame实现ALS算法库使用，构建训练电影推荐模型
 */
object SparkRmdAls {
	
	def main(args: Array[String]): Unit = {
		
		val spark = SparkSession.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[4]")
			.config("spark.sql.shuffle.partitions", "4")
			.getOrCreate()
		import org.apache.spark.sql.functions._
		import spark.implicits._
		
		// 自定义Schame信息
		val mlSchema = StructType(
			Array(
				StructField("userId", IntegerType, nullable = true),
				StructField("movieId", IntegerType, nullable = true),
				StructField("rating", DoubleType, nullable = true),
				StructField("timestamp", LongType, nullable = true)
			)
		)
		// TODO: 读取电影评分数据，数据格式为TSV格式
		val rawRatingsDF: DataFrame = spark.read
			.option("sep", "\t")
			.schema(mlSchema)
			.csv("datas/als/ml-100k/u.data")
		
		//rawRatingsDF.printSchema()
		/**
		 * root
		 * |-- userId: integer (nullable = true)
		 * |-- movieId: integer (nullable = true)
		 * |-- rating: double (nullable = true)
		 * |-- timestamp: long (nullable = true)
		 */
		//rawRatingsDF.show(10, truncate = false)
		
		// TODO: ALS 算法实例对象
		val als = new ALS() // def this() = this(Identifiable.randomUID("als"))
			// 设置迭代的最大次数
			.setMaxIter(10)
			// 设置特征数
			.setRank(10)
			// 显式评分
			.setImplicitPrefs(false)
			// 设置Block的数目， be partitioned into in order to parallelize computation (默认值: 10).
			.setNumItemBlocks(4).setNumUserBlocks(4)
			// 设置 用户ID:列名、产品ID:列名及评分:列名
			.setUserCol("userId")
			.setItemCol("movieId")
			.setRatingCol("rating")
		// TODO: 使用数据训练模型
		val mlAlsModel: ALSModel = als.fit(rawRatingsDF)
		
		// TODO: 用户-特征 因子矩阵
		val userFeaturesDF: DataFrame = mlAlsModel.userFactors
		userFeaturesDF.show(5, truncate = false)
		// TODO: 产品-特征 因子矩阵
		val itemFeaturesDF: DataFrame = mlAlsModel.itemFactors
		itemFeaturesDF.show(5, truncate = false)
		
		/**
		 * 推荐模型：
		 * 1、预测用户对物品评价，喜好
		 * 2、为用户推荐物品（推荐电影）
		 * 3、为物品推荐用户（推荐用户）
		 */
		// TODO: 预测 用户对产品（电影）评分
		val predictRatingsDF: DataFrame = mlAlsModel
			// 设置 用户ID:列名、产品ID:列名
			.setUserCol("userId").setItemCol("movieId")
			.setPredictionCol("predictRating") // 默认列名为 prediction
			.transform(rawRatingsDF)
		predictRatingsDF.show(5, truncate = false)
		
		/*
		 TODO：实际项目中，构建出推荐模型以后，获取给用户推荐商品或者给物品推荐用户，往往存储至NoSQL数据库
		    1、数据量表较大
		    2、查询数据比较快
		    可以是Redis内存数据，MongoDB文档数据，HBase面向列数据，存储Elasticsearch索引中
		 */
		// TODO: 为用户推荐10个产品（电影）
		// max number of recommendations for each user
		val userRecs: DataFrame = mlAlsModel.recommendForAllUsers(10)
		userRecs.show(5, truncate = false)
		// 查找 某个用户推荐的10个电影，比如用户:196     将结果保存Redis内存数据库，可以快速查询检索
		
		// TODO: 为产品（电影）推荐10个用户
		// max number of recommendations for each item
		val movieRecs: DataFrame = mlAlsModel.recommendForAllItems(10)
		movieRecs.show(5, truncate = false)
		// 查找 某个电影推荐的10个用户，比如电影:242
		
		// TODO: 模型的评估
		val evaluator = new RegressionEvaluator()
			.setLabelCol("rating")
			.setPredictionCol("predictRating")
			.setMetricName("rmse")
		val rmse = evaluator.evaluate(predictRatingsDF)
		println(s"Root-mean-square error = $rmse")
		
		// TODO: 模型的保存
		mlAlsModel.save("datas/als/mlalsModel")
		
		// TODO: 加载模型
		val loadMlAlsModel: ALSModel = ALSModel.load( "datas/als/mlalsModel")
		loadMlAlsModel.recommendForAllItems(10).show(5, truncate = false)
		
		// 应用结束，关闭资源
		Thread.sleep(1000000)
		spark.stop()
	}
	
}
