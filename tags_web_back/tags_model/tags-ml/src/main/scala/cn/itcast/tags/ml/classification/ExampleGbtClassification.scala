package cn.itcast.tags.ml.classification

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel

/**
 * Spark ML官方案例，基于梯度提升树分类算法
 * 	    http://spark.apache.org/docs/2.2.0/ml-classification-regression.html#gradient-boosted-tree-classifier
 */
object ExampleGbtClassification {
	
	def main(args: Array[String]): Unit = {
		
		// 构建SparkSession实例对象，通过建造者模式创建
		import org.apache.spark.sql.SparkSession
		val spark: SparkSession = {
			SparkSession
				.builder()
				.appName(this.getClass.getSimpleName.stripSuffix("$"))
				.master("local[3]")
				.config("spark.sql.shuffle.partitions", "3")
				.getOrCreate()
		}
		// 导入隐式转换和函数库
		import spark.implicits._
		import org.apache.spark.sql.functions._
		
		// TODO: 1. 加载数据、数据过滤与基本转换
		val datasDF: DataFrame = spark.read
			.format("libsvm")
			.load("datas/mllib/sample_libsvm_data.txt")
		
		// TODO: 2. 数据准备：特征工程（提取、转换与选择）
		// 将标签数据转换为从0开始下标索引
		val labelIndexer: StringIndexerModel = new StringIndexer()
			.setInputCol("label")
			.setOutputCol("label_index")
			.fit(datasDF)
		val indexerDF = labelIndexer.transform(datasDF)
		
		// 自动识别特征数据中属于类别特征的字段，进行索引转换，决策树中使用类别特征更加好
		val featureIndexer: VectorIndexerModel = new VectorIndexer()
			.setInputCol("features")
			.setOutputCol("index_features")
			.setMaxCategories(4)
			.fit(indexerDF)
		val dataframe = featureIndexer.transform(indexerDF)
		
		// 划分数据集：训练数据集和测试数据集
		val Array(trainingDF, testingDF) = dataframe.randomSplit(Array(0.8, 0.2))
		trainingDF.persist(StorageLevel.MEMORY_AND_DISK).count()
		
		// TODO: 3. 使用算法和数据构建模型：算法参数
		val gbt: GBTClassifier = new GBTClassifier()
			.setLabelCol("label_index")
			.setFeaturesCol("index_features")
			// 设置超参数
    		.setMaxIter(10)
			.setStepSize(0.1) // 学习率，(0, 1]之间，默认值为1
			.setSubsamplingRate(1.0) // 每次训练决策树数据集占比，默认为1.0
    		//.setImpurity("variance")
    		//.setLossType("logistic")
			// 树的参数
    		.setImpurity("gini")
			.setMaxDepth(5)
			.setMaxBins(32)
		val gbtModel: GBTClassificationModel = gbt.fit(trainingDF)
		
		// TODO: 4. 模型评估与参数调优
		val predictionDF: DataFrame = gbtModel.transform(testingDF)
		predictionDF
			.select("prediction", "label_index")
			.show(50, truncate = false)
		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("label_index")
			.setPredictionCol("prediction")
			.setMetricName("accuracy")
		val accuracy = evaluator.evaluate(predictionDF)
		println("Test Error = " + (1.0 - accuracy))
		
		Thread.sleep(10000000)
		// 应用结束，关闭资源
		spark.stop()
	}
	
}