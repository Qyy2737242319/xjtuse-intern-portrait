package cn.itcast.tags.ml.clustering

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.immutable

/**
 * 针对鸢尾花数据集进行聚类，使用KMeans算法，采用肘部法则Elbow获取K的值，使用轮廓系数评估模型
 */
object IrisElbowClustering {
	
	def main(args: Array[String]): Unit = {
		
		// 构建SparkSession实例对象
		val spark: SparkSession = SparkSession.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[4]")
			.config("spark.sql.shuffle.partitions", "4")
			.getOrCreate()
		import spark.implicits._
		import org.apache.spark.sql.functions._
		
		// 1. 加载鸢尾花数据，使用libsvm格式
		val irisDF: DataFrame = spark.read
			.format("libsvm")
			.load("datas/iris_kmeans.txt")
		/*
		root
		 |-- label: double (nullable = true)
		 |-- features: vector (nullable = true)
		 */
		//irisDF.printSchema()
		//irisDF.show(10, truncate = false)
		
		// 2. 设置不同K，从2开始到6，采用肘部法确定K值
		val values: immutable.Seq[(Int, KMeansModel, String, Double)] = (2 to 6).map{ k =>
			// a. 创建KMeans算法实例对象，设置参数值
			val kmeans: KMeans = new KMeans()
				// 设置输入特征列名称和输出列的名名称
				.setFeaturesCol("features")
				.setPredictionCol("prediction")
				.setK(k)
				.setMaxIter(50)
				.setInitMode("k-means||")
    			//.setDistanceMeasure("euclidean")
    			.setDistanceMeasure("cosine")
			
			// b. 应用数据集训练模型, 获取转换器
			val kmeansModel: KMeansModel = kmeans.fit(irisDF)
			
			// c. 模型预测
			val predictionDF: DataFrame = kmeansModel.transform(irisDF)
			
			// 统计出各个类簇中数据个数
			// 0-> 12,  1->14, 2 -> 23
			val clusterNumber: String = predictionDF
				.groupBy($"prediction").count()
    			.select($"prediction", $"count")
    			.as[(Int, Long)]
    			.rdd.collectAsMap()
    			.toMap.mkString(", ")
			
			// d. 模型评估
			val evaluator: ClusteringEvaluator = new ClusteringEvaluator()
				// 设置预测值
				.setPredictionCol("prediction")
				// 评估指标：轮廓系数
				.setMetricName("silhouette")
				// 采用欧式距离计算距离
				//.setDistanceMeasure("squaredEuclidean")
				.setDistanceMeasure("cosine")
			val scValue: Double = evaluator.evaluate(predictionDF)
			
			// e. 返回四元组
			(k, kmeansModel, clusterNumber, scValue)
		}
		
		// 3. 当K为不同值时，训练模型，使用轮廓系数评估
		/*
			(2,kmeans_2a49caed35bd,1 -> 97, 0 -> 53,0.8501515983265806)
			(3,kmeans_96bb2594f1dc,2 -> 39, 1 -> 50, 0 -> 61,0.7342113066202725)
			(4,kmeans_5cfd34c5f44c,2 -> 28, 1 -> 50, 3 -> 43, 0 -> 29,0.6748661728223084)
			(5,kmeans_24948112248e,0 -> 23, 1 -> 33, 2 -> 30, 3 -> 47, 4 -> 17,0.5593200358940349)
			(6,kmeans_a7b92dbdea34,0 -> 30, 5 -> 18, 1 -> 19, 2 -> 47, 3 -> 23, 4 -> 13,0.5157126401818913)
			
			使用余弦距离计算数据之间距离
			(2,kmeans_8bc0305c6a30,1 -> 50, 0 -> 100,0.9579554849242657)
			(3,kmeans_4c6ac2a5c61c,2 -> 46, 1 -> 50, 0 -> 54,0.7484647230660575)
			(4,kmeans_10dbb33d5f8b,2 -> 46, 1 -> 19, 3 -> 31, 0 -> 54,0.5754341193280768)
			(5,kmeans_33807d6cff31,0 -> 27, 1 -> 50, 2 -> 23, 3 -> 28, 4 -> 22,0.6430770644178772)
			(6,kmeans_89d5c891319b,0 -> 24, 5 -> 21, 1 -> 29, 2 -> 43, 3 -> 15, 4 -> 18,0.4512255960897416)
		 */
		values.foreach(println)
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
