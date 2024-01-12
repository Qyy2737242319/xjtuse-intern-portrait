package cn.itcast.tags.ml.features

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, LogisticRegressionSummary}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructType}

/**
 * 读取鸢尾花数据集数据，封装特征值features和标签处理label
 *
 * TODO: 全部基于DataFrame API实现
 */
object IrisFeaturesDemo {
	
	def main(args: Array[String]): Unit = {
	
		// 构建SparkSession实例对象
		val spark: SparkSession = SparkSession.builder()
    		.appName(this.getClass.getSimpleName.stripSuffix("$"))
    		.master("local[4]")
    		.config("spark.sql.shuffle.partitions", 4)
    		.getOrCreate()
		import spark.implicits._
	
		
		// TODO: 1. 加载鸢尾花数据集iris.data，属于csv文件
		val irisSchema: StructType = new StructType()
    		.add("sepal_length", DoubleType, nullable = true)
    		.add("sepal_width", DoubleType, nullable = true)
    		.add("petal_length", DoubleType, nullable = true)
    		.add("petal_width", DoubleType, nullable = true)
    		.add("class", StringType, nullable = true)
		val rawIrisDF: DataFrame = spark.read
            .option("sep", ",")
			// 当CSV文件首行不是列名称时，自定义Schema
            .option("header", "false")
            .option("inferSchema", "false")
    		.schema(irisSchema)
			.csv("datas/iris/iris.data")
		//rawIrisDF.printSchema()
		//rawIrisDF.show(10, truncate = false)
	
		// TODO: step1 -> 将 萼片长度、宽度及花瓣长度、宽度 封装值 特征features向量中
		val assembler = new VectorAssembler()
			.setInputCols(rawIrisDF.columns.dropRight(1))
			.setOutputCol("features") // 添加一列，类型为向量
		val df1: DataFrame = assembler.transform(rawIrisDF)
		///df1.printSchema()
		//df1.show(10, truncate = false)
		
		// TODO: step2 -> 转换类别字符串数据为数值数据
		val indexer: StringIndexer = new StringIndexer()
			.setInputCol("class") // 对哪列数据进行索引化
			.setOutputCol("label") // 数据索引化后列名
		val df2: DataFrame = indexer
			.fit(df1) // fit方法表示调用函数，传递DataFrame，获取模型
			.transform(df1)
		/*
		root
		 |-- sepal_length: double (nullable = true)
		 |-- sepal_width: double (nullable = true)
		 |-- petal_length: double (nullable = true)
		 |-- petal_width: double (nullable = true)
		 |-- class: string (nullable = true)
		 |-- features: vector (nullable = true) // 特征 x
		 |-- label: double (nullable = false) // 标签 y
		 算法：y = kx + b
		 */
		//df2.printSchema()
		//df2.show(150, truncate = false)
		
		// TODO: step3 -> 将特征数据features进行标准化处理转换，使用StandardScaler
		/*
			机器学习核心三要素：  数据（指的就是特征features） + 算法 = 模型（最佳）
				调优中，最重要的就是特征数据features，如果特征数据比较好，处理恰当，可能得到较好模型
			TODO：在实际开发中，特征数据features需要进行各个转换操作，比如正则化、归一化或标准化等等
				为什么需要对特征数据features进行归一化等操作呢？？？？ 原因在于不同维度特征值，值的范围跨度不一样，导致模型异常
		 */
		val scaler: StandardScaler = new StandardScaler()
			.setInputCol("features")
			.setOutputCol("scale_features")
			.setWithStd(true) // 使用标准差缩放
			.setWithMean(false) // 不适用平均值缩放
		val irisDF: DataFrame = scaler
			.fit(df2)
			.transform(df2)
		//irisDF.printSchema()
		//irisDF.show(10, truncate = false)
		
		// TODO: 2. 选择一个分类算法，构建分类模型
		/*
		 分类算法属于最多算法，比如如下分类算法：
		    1. 决策树（DecisionTree）分类算法
		    2. 朴素贝叶斯（Naive Bayes）分类算法，适合构建文本数据特征分类，比如垃圾邮件，情感分析
		    3. 逻辑回归（Logistics Regression）分类算法
		    4. 线性支持向量机（Linear SVM）分类算法
		    5. 神经网络相关分类算法，比如多层感知机算法 -> 深度学习算法
		    6. 集成融合算法：随机森林（RF）分类算法、梯度提升树（GBT）算法
		 */
		val lr: LogisticRegression = new LogisticRegression()
			// 设置特征值列名称和标签列名称
			.setFeaturesCol("scale_features") // x -> 特征
			.setLabelCol("label") // y -> 标签
			// 每个算法都有自己超参数要设置，比较关键，合理设置，获取较好的模型
			.setMaxIter(20) // 模型训练迭代次数
			.setFamily("multinomial") // 设置分类属于二分类（标签label只有2个值）还是多分类（标签label有多个值，大于2）
			.setStandardization(true) // 是否对特征数据进行标准化
			.setRegParam(0) // 正则化参数，优化
			.setElasticNetParam(0) // 弹性化参数，优化
		// TODO: 将数据应用到算法中，训练模型
		val lrModel: LogisticRegressionModel = lr.fit(irisDF)
		
		// TODO: 评估模型到底如何
		val summary: LogisticRegressionSummary = lrModel.summary
		// 准确度：accuracy: 0.9733333333333334
		println(s"accuracy: ${summary.accuracy}")
		// 精确度：precision: 1.0, 0.96, 0.96， 针对每个类别数据预测准确度
		println(s"precision: ${summary.precisionByLabel.mkString(", ")}")
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
