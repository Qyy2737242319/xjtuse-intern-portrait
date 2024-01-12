package cn.itcast.tags.ml.rs.rdd

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 使用MovieLens 电影评分数据集，调用Spark MLlib 中协同过滤推荐算法ALS建立推荐模型：
 * -a. 预测 用户User 对 某个电影Product 评价
 * -b. 为某个用户推荐10个电影Products
 * -c. 为某个电影推荐10个用户Users
 */
object SparkAlsRmdMovie {
	
	def main(args: Array[String]): Unit = {
		
		// TODO: 1. 构建SparkContext实例对象
		val sc: SparkContext = {
			// a. 创建SparkConf对象，设置应用相关配置
			val sparkConf = new SparkConf()
				.setMaster("local[2]")
				.setAppName(this.getClass.getSimpleName.stripSuffix("$"))
			// b. 创建SparkContext
			val context = SparkContext.getOrCreate(sparkConf)
			// 设置检查点目录
			context.setCheckpointDir(s"datas/ckpt/als-ml-${System.nanoTime()}")
			// c. 返回
			context
		}
		
		// TODO: 2. 读取 电影评分数据
		val rawRatingsRDD: RDD[String] = sc.textFile("datas/als/ml-100k/u.data")
		println(s"Count = ${rawRatingsRDD.count()}")
		println(s"First: ${rawRatingsRDD.first()}")
		
		// TODO: 3. 数据转换，构建RDD[Rating]
		val ratingsRDD: RDD[Rating] = rawRatingsRDD
			// 过滤不合格的数据
			.filter(line => null != line && line.split("\\t").length == 4)
			.map{line =>
				// 字符串分割
				val Array(userId, movieId, rating, _) = line.split("\\t")
				// 返回Rating实例对象
				Rating(userId.toInt, movieId.toInt, rating.toDouble)
			}
		// 划分数据集为训练数据集和测试数据集
		val Array(trainRatings, testRatings) = ratingsRDD.randomSplit(Array(0.9, 0.1))
		
		// TODO： 4. 调用ALS算法中显示训练函数训练模型
		// 迭代次数为20，特征数为10
		val alsModel: MatrixFactorizationModel = ALS.train(
			ratings = trainRatings, // 训练数据集
			rank = 10, // 特征数rank
			iterations = 20 // 迭代次数
		)
		
		// TODO: 5. 获取模型中两个因子矩阵
		/**
		 * 获取模型MatrixFactorizationModel就是里面包含两个矩阵：
		 * -a. 用户因子矩阵
		 *         alsModel.userFeatures
		 * -b. 产品因子矩阵
		 *         alsModel.productFeatures
		 */
		// userId -> Features
		val userFeatures: RDD[(Int, Array[Double])] = alsModel.userFeatures
		userFeatures.take(10).foreach{tuple =>
			println(tuple._1 + " -> " + tuple._2.mkString(","))
		}
		println("=======================================================")
		// productId -> Features
		val productFeatures: RDD[(Int, Array[Double])] = alsModel.productFeatures
		productFeatures.take(10).foreach{
			tuple => println(tuple._1 + " -> " + tuple._2.mkString(","))
		}
		
		
		// TODO: 6. 模型评估，使用RMSE评估模型，值越小，误差越小，模型越好
		// 6.1 转换测试数据集格式RDD[((userId, ProductId), rating)]
		val actualRatingsRDD: RDD[((Int, Int), Double)] = testRatings.map{tuple =>
			((tuple.user, tuple.product), tuple.rating)
		}
		// 6.2 使用模型对测试数据集预测电影评分
		val predictRatingsRDD: RDD[((Int, Int), Double)] = alsModel
			// 依据UserId和ProductId预测评分
			.predict(actualRatingsRDD.map(_._1))
			// 转换数据格式RDD[((userId, ProductId), rating)]
			.map(tuple => ((tuple.user, tuple.product), tuple.rating))
		// 6.3 合并预测值与真实值
		val predictAndActualRatingsRDD: RDD[((Int, Int), (Double, Double))] = predictRatingsRDD.join(actualRatingsRDD)
		// 6.4 模型评估，计算RMSE值
		val metrics = new RegressionMetrics(predictAndActualRatingsRDD.map(_._2))
		println(s"RMSE = ${metrics.rootMeanSquaredError}")
		
		
		// TODO 7. 推荐与预测评分
		// 7.1 预测某个用户对某个产品的评分  def predict(user: Int, product: Int): Double
		val predictRating: Double = alsModel.predict(196, 242)
		println(s"预测用户196对电影242的评分：$predictRating")
		
		println("----------------------------------------")
		// 7.2 为某个用户推荐十部电影  def recommendProducts(user: Int, num: Int): Array[Rating]
		val rmdMovies: Array[Rating] = alsModel.recommendProducts(196, 10)
		rmdMovies.foreach(println)
		
		println("----------------------------------------")
		// 7.3 为某个电影推荐10个用户  def recommendUsers(product: Int, num: Int): Array[Rating]
		val rmdUsers = alsModel.recommendUsers(242, 10)
		rmdUsers.foreach(println)
		
		// TODO: 8. 将训练得到的模型进行保存，以便后期加载使用进行推荐
		val modelPath = s"datas/als/ml-als-model-" + System.nanoTime()
		alsModel.save(sc, modelPath)
		
		// TODO: 9. 从文件系统中记载保存的模型，用于推荐预测
		val loadAlsModel: MatrixFactorizationModel = MatrixFactorizationModel
			.load(sc, modelPath)
		// 使用加载预测
		val loaPredictRating: Double = loadAlsModel.predict(196, 242)
		println(s"加载模型 -> 预测用户196对电影242的评分：$loaPredictRating")
		
		// 为了WEB UI监控，线程休眠
		Thread.sleep(10000000)
		
		// 关闭资源
		sc.stop()
	}
	
}
