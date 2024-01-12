package cn.itcast.tags.ml.clustering

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 使用鸢尾花数据集，基于KMeans聚类算法构建聚类模型，并对模型进行评估
 */
object IrisClusterDemo {
	
	def main(args: Array[String]): Unit = {
		
		val spark = SparkSession.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[2]")
			.config("spark.sql.shuffle.partitions", "2")
			.getOrCreate()
		import org.apache.spark.sql.functions._
		import spark.implicits._
		
		// 1. 读取鸢尾花数据集
		val irisDF: DataFrame = spark.read
			.format("libsvm")
    		.option("numFeatures", "4")
			.load("datas/iris_kmeans.txt")
		/*
		root
		 |-- label: double (nullable = true)
		 |-- features: vector (nullable = true)
		 */
		//irisDF.printSchema()
		/*
			+-----+-------------------------------+
			|label|features                       |
			+-----+-------------------------------+
			|1.0  |(4,[0,1,2,3],[5.1,3.5,1.4,0.2])|
			|1.0  |(4,[0,1,2,3],[4.9,3.0,1.4,0.2])|
			|1.0  |(4,[0,1,2,3],[4.7,3.2,1.3,0.2])|
			|1.0  |(4,[0,1,2,3],[4.6,3.1,1.5,0.2])|
			|1.0  |(4,[0,1,2,3],[5.0,3.6,1.4,0.2])|
			|1.0  |(4,[0,1,2,3],[5.4,3.9,1.7,0.4])|
		 */
		//irisDF.show(10, truncate = false)
		
		// 2. 构建KMeans算法-> 模型学习器实例对象
		val kmeans: KMeans = new KMeans()
			// 设置输入特征列名称和输出列的名名称
			.setFeaturesCol("features")
			.setPredictionCol("prediction")
			// 设置K值为3
			.setK(3)
			// 设置最大的迭代次数
			.setMaxIter(20)
			// 设置KMeans算法底层: random 、k-means||
    		.setInitMode("k-means||")
		
		// 3. 应用数据集训练模型, 获取转换器
		val kMeansModel: KMeansModel = kmeans.fit(irisDF)
		// 获取聚类的簇中心点
		/*
			[5.88360655737705,2.7409836065573776,4.388524590163936,1.4344262295081969]
			[5.005999999999999,3.4180000000000006,1.4640000000000002,0.2439999999999999]
			[6.853846153846153,3.0769230769230766,5.715384615384615,2.053846153846153]
		 */
		kMeansModel.clusterCenters.foreach(println)
		
		// 4. 模型评估
		val wssse: Double = kMeansModel.computeCost(irisDF)
		// WSSSE = 78.94506582597637
		println(s"WSSSE = ${wssse}")
		
		// 5. 使用模型预测
		val predictionDF: DataFrame = kMeansModel.transform(irisDF)
		predictionDF.show(150, truncate = false)
		predictionDF
			.groupBy($"label", $"prediction")
			.count()
			.show(150, truncate = false)
		
		
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
