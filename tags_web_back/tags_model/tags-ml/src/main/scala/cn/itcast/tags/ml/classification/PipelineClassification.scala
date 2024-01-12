package cn.itcast.tags.ml.classification

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object PipelineClassification {
	
	def main(args: Array[String]): Unit = {
		
		val spark = SparkSession.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[4]")
			.getOrCreate()
		import org.apache.spark.sql.functions._
		import spark.implicits._
		
		// 1. 加载数据
		val dataframe: DataFrame = spark.read
			.format("libsvm")
			.load("datas/mllib/sample_libsvm_data.txt")
		//dataframe.printSchema()
		//dataframe.show(10, truncate = false)
		
		// 划分数据集：训练数据和测试数据
		val Array(trainingDF, testingDF) = dataframe.randomSplit(Array(0.8, 0.2))
		
		// 2. 特征工程：特征提取、特征转换及特征选择
		// 2.1. 将标签值label，转换为索引，从0开始，到 K-1
		val labelIndexer: StringIndexerModel = new StringIndexer()
			.setInputCol("label")
			.setOutputCol("index_label")
			.fit(dataframe)
		val df1: DataFrame = labelIndexer.transform(dataframe)
		
		// 2.2. 对类别特征数据进行特殊处理, 当每列的值的个数小于等于设置K，那么此列数据被当做类别特征，自动进行索引转换
		val featureIndexer: VectorIndexerModel = new VectorIndexer()
			.setInputCol("features")
			.setOutputCol("index_features")
			// TODO: 表示哪些特征列为类别特征列，并且将特征列的特征值进行索引化转换操作
			.setMaxCategories(4) // 类别特征最大类别个数
			.fit(df1)
		val df2: DataFrame = featureIndexer.transform(df1)
		
		// 3. 使用决策树算法构建分类模型
		val dtc: DecisionTreeClassifier = new DecisionTreeClassifier()
			.setLabelCol("index_label")
			.setFeaturesCol("index_features")
			// 设置决策树算法相关超参数
			.setImpurity("gini") // 也可以是香农熵：entropy
			.setMaxDepth(5)
			.setMaxBins(32) // 此值必须大于等于类别特征类别个数
		
		// TODO: 4. 构建Pipeline管道，设置Stage，每个Stage要么是算法（模型学习器Estimator），要么是模型（转换器Transformer）
		val pipeline: Pipeline = new Pipeline()
			// 设置Stage，依赖顺序
    		.setStages(
			    Array(labelIndexer, featureIndexer, dtc)
		    )
		// 应用数据集，训练管道模型
		val pipelineModel: PipelineModel = pipeline.fit(trainingDF)
		// TODO： 获取决策树分类模型
		val dtcModel: DecisionTreeClassificationModel = pipelineModel
			.stages(2)
			.asInstanceOf[DecisionTreeClassificationModel]
		println(dtcModel.toDebugString)
		
		// 4. 模型评估
		val predictionDF: DataFrame = pipelineModel.transform(testingDF)
		predictionDF.printSchema()
		predictionDF
			.select($"index_label", $"probability", $"prediction")
			.show(20, truncate = false)
		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("index_label")
			.setPredictionCol("prediction")
			.setMetricName("accuracy")
		val accuracy: Double = evaluator.evaluate(predictionDF)
		println(s"Accuracy = $accuracy")
		
		spark.stop()
	}
	
}
