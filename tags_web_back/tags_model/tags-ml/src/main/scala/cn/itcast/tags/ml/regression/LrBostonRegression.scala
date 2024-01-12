package cn.itcast.tags.ml.regression

import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.storage.StorageLevel

/**
 * 使用线性回归算法对波士顿房价数据集构建回归模型，评估模型性能
 *      TODO： 波士顿房价数据集，共506条数据，13个属性（特征值，features），1个房价（预测值，target）
 */
object LrBostonRegression {
	
	def main(args: Array[String]): Unit = {
		// 构建SparkSession实例对象，通过建造者模式创建
		val spark: SparkSession = SparkSession
			.builder()
			.appName(this.getClass.getSimpleName.stripSuffix("$"))
			.master("local[3]")
			.config("spark.sql.shuffle.partitions", "3")
			.getOrCreate()
		import spark.implicits._
		
		
		// TODO: 1. 加载波士顿房价数据集
		val bostonPriceDF: Dataset[String] = spark.read
			.textFile("datas/housing/housing.data")
			.filter(line => null != line && line.trim.split("\\s+").length == 14)
		
		
		// TODO: 2. 获取特征features和标签label
		val bostonDF: DataFrame = bostonPriceDF
			.mapPartitions{iter =>
				iter.map{line =>
					val parts: Array[String] = line.trim.split("\\s+")
					// 获取标签label
					val label: Double = parts(parts.length - 1).toDouble
					// 获取特征features
					val values: Array[Double] = parts.dropRight(1).map(_.toDouble)
					val features: linalg.Vector = Vectors.dense(values)
					// 返回二元组
					(features, label)
				}
			}
			// 调用toDF函数，指定列名称
			.toDF("features", "label")
		//bostonDF.printSchema()
		//bostonDF.show(20, truncate = false)
		
		// TODO: 需要将数据集划分为训练数据集和测试数据集
		val Array(trainingDF, testingDF) = bostonDF.randomSplit(Array(0.8, 0.2), seed = 123L)
		trainingDF.persist(StorageLevel.MEMORY_AND_DISK).count() // 触发缓存
		
		// TODO: 3. 特征数据转换处理（归一化）等
		// 由于线性回归算法，默认情况下，对特征features数据进行标准化处理转换，所以此处不再进行处理
		
		// TODO: 4. 创建线性回归算法实例对象，设置相关参数，并且应用数据训练模型
		val lr: LinearRegression = new LinearRegression()
			// 设置特征列和标签列名称
    		.setFeaturesCol("features")
			.setLabelCol("label")
    		// 是否对特征数据进行标准化转换处理
			.setStandardization(true)
			// 设置算法底层求解方式，要么是最小二乘法（正规方程normal），要么是拟牛顿法（l-bfgs）
			.setSolver("auto")
			// 设置算法相关超参数的值
			.setMaxIter(20)
			.setRegParam(1).setElasticNetParam(0.4)
		val lrModel: LinearRegressionModel = lr.fit(trainingDF)
		
		// TODO: 5. 模型评估，查看模型如何
		// Coefficients：斜率，就是k    Intercept：截距，就是b
		println(s"Coefficients: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")
		
		val trainingSummary = lrModel.summary
		println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
		
		// TODO 6. 模型预测
		lrModel.transform(testingDF).show(10, truncate = false)
		
		// 应用结束，关闭资源
		spark.stop()
	}
	
}
