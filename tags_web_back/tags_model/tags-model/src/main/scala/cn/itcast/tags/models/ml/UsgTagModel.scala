package cn.itcast.tags.models.ml

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.{MLModelTools, TagTools}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
 * 挖掘类型标签模型开发：用户购物性别标签模型
 */
class UsgTagModel extends AbstractModel("用户购物性别USG", ModelType.ML){
	/*
	378	用户购物性别
		379	男		0
		380	女		1
		381	中性		-1
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session: SparkSession = businessDF.sparkSession
		import session.implicits._
		
		/*
			root
			 |-- cordersn: string (nullable = true)
			 |-- ogcolor: string (nullable = true)
			 |-- producttype: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(10, truncate = false)
		
		//tagDF.printSchema()
		/*
			+---+----+----+-----+
			|id |name|rule|level|
			+---+----+----+-----+
			|379|男   |0   |5    |
			|380|女   |1   |5    |
			|381|中性  |-1  |5    |
			+---+----+----+-----+
		 */
		//tagDF.filter($"level".equalTo(5)).show(10, truncate = false)
		
		
		// 1. 获取订单表数据tbl_tag_orders，与订单商品表数据关联获取会员ID
		val ordersDF: DataFrame = spark.read
			.format("hbase")
			.option("zkHosts", "bigdata-cdh01.itcast.cn")
			.option("zkPort", "2181")
			.option("hbaseTable", "tbl_tag_orders")
			.option("family", "detail")
			.option("selectFields", "memberid,ordersn")
			.load()
		//ordersDF.printSchema()
		//ordersDF.show(10, truncate = false)
		
		// 2. 加载维度表数据：tbl_dim_colors（颜色）、tbl_dim_products（产品）
		// 2.1 加载颜色维度表数据
		val colorsDF: DataFrame = {
			spark.read
				.format("jdbc")
				.option("driver", "com.mysql.jdbc.Driver")
				.option("url",
					"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC")
				.option("dbtable", "profile_tags.tbl_dim_colors")
				.option("user", "root")
				.option("password", "123456")
				.load()
		}
		/*
			root
			 |-- id: integer (nullable = false)
			 |-- color_name: string (nullable = true)
		 */
		//colorsDF.printSchema()
		//colorsDF.show(30, truncate = false)
		
		// 2.2. 构建颜色WHEN语句
		val colorColumn: Column = {
			// 声明变量
			var colorCol: Column = null
			colorsDF
				.as[(Int, String)].rdd
				.collectAsMap()
				.foreach{case (colorId, colorName) =>
					if(null == colorCol){
						colorCol = when($"ogcolor".equalTo(colorName), colorId)
					}else{
						colorCol = colorCol.when($"ogcolor".equalTo(colorName), colorId)
					}
				}
			colorCol = colorCol.otherwise(0).as("color")
			//  返回
			colorCol
		}
		
		// 2.3 加载商品维度表数据
		val productsDF: DataFrame = {
			spark.read
				.format("jdbc")
				.option("driver", "com.mysql.jdbc.Driver")
				.option("url",
					"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC")
				.option("dbtable", "profile_tags.tbl_dim_products")
				.option("user", "root")
				.option("password", "123456")
				.load()
		}
		/*
			root
			 |-- id: integer (nullable = false)
			 |-- product_name: string (nullable = true)
		 */
		//productsDF.printSchema()
		///productsDF.show(30, truncate = false)
		
		// 2.4. 构建商品类别WHEN语句
		var productColumn: Column = {
			// 声明变量
			var productCol: Column = null
			productsDF
				.as[(Int, String)].rdd
				.collectAsMap()
				.foreach{case (productId, productName) =>
					if(null == productCol){
						productCol = when($"producttype".equalTo(productName), productId)
					}else{
						productCol = productCol.when($"producttype".equalTo(productName), productId)
					}
				}
			productCol = productCol.otherwise(0).as("product")
			// 返回
			productCol
		}
		
		// 2.5. 根据运营规则标注的部分数据
		val labelColumn: Column = {
			when($"ogcolor".equalTo("樱花粉")
				.or($"ogcolor".equalTo("白色"))
				.or($"ogcolor".equalTo("香槟色"))
				.or($"ogcolor".equalTo("香槟金"))
				.or($"productType".equalTo("料理机"))
				.or($"productType".equalTo("挂烫机"))
				.or($"productType".equalTo("吸尘器/除螨仪")), 1) //女
				.otherwise(0)//男
				.alias("label")//决策树预测label
		}
		
		// 3. 关联订单数据，颜色维度和商品类别维度数据
		val goodsDF: DataFrame = businessDF
				// 关联订单数据：
	            .join(ordersDF, businessDF("cordersn") === ordersDF("ordersn"))
				// 选择所需字段，使用when判断函数
	            .select(
				    $"memberid".as("userId"), //
				    colorColumn, // 颜色ColorColumn
				    productColumn, // 产品类别ProductColumn
				    // 依据规则标注商品性别
				    labelColumn
			    )
		//goodsDF.printSchema()
		//goodsDF.show(100, truncate = false)
		
		// TODO: 直接使用标注数据，给用户打标签
		//val predictionDF: DataFrame = goodsDF.select($"userId", $"label".as("prediction"))
		
		// TODO: 实际上，需要使用标注数据（给每个用户购买每个商品打上性别字段），构建算法模型，并使用模型预测值性别字段值
		//val featuresDF: DataFrame = featuresTransform(goodsDF)
		//val dtcModel: DecisionTreeClassificationModel = trainModel(featuresDF)
		//val predictionDF: DataFrame = dtcModel.transform(featuresDF)
		
		// TODO: 使用数据集训练管道模型PipelineModel
		//val pipelineModel: PipelineModel = trainPipelineModel(goodsDF)
		//val predictionDF: DataFrame = pipelineModel.transform(goodsDF)
		
		// TODO: 使用训练验证分割方式，设置超参数训练处最佳模型
		//val pipelineModel: PipelineModel = trainBestModel(goodsDF)
		//val predictionDF: DataFrame = pipelineModel.transform(goodsDF)
		
		// TODO: 使用交叉验证方式，设置超参数训练获取最佳模型
		/*
			当模型存在时，直接加载模型；如果不存在，训练获取最佳模型，并保存
		 */
		val pipelineModel: PipelineModel = MLModelTools.loadModel(
			goodsDF, "usg", this.getClass
		).asInstanceOf[PipelineModel]
		val predictionDF: DataFrame = pipelineModel.transform(goodsDF)
		
		// 4. 按照用户ID分组，统计每个用户购物男性或女性商品个数及占比
		val genderDF: DataFrame = predictionDF
    		.groupBy($"userId")
    		.agg(
			    count($"userId").as("total"), // 某个用户购物商品总数
			    // 判断label为0时，表示为男性商品，设置为1，使用sum函数累加
			    sum(
				    when($"prediction".equalTo(0), 1).otherwise(0)
			    ).as("maleTotal"),
			    // 判断label为1时，表示为女性商品，设置为1，使用sum函数累加
			    sum(
				    when($"prediction".equalTo(1), 1).otherwise(0)
			    ).as("femaleTotal")
		    )
		/*
		root
			 |-- userId: string (nullable = true)
			 |-- total: long (nullable = false)
			 |-- maleTotal: long (nullable = true)
			 |-- femaleTotal: long (nullable = true)
		 */
		//genderDF.printSchema()
		//genderDF.show(10, truncate = false)
		
		// 5. 计算标签
		// 5.1 获取属性标签：tagRule和tagName
		val rulesMap: Map[String, String] = TagTools.convertMap(tagDF)
		val rulesMapBroadcast: Broadcast[Map[String, String]] = session.sparkContext.broadcast(rulesMap)
		
		// 5.2 自定义UDF函数，计算占比，确定标签值
		// 对每个用户，分别计算男性商品和女性商品占比，当占比大于等于0.6时，确定购物性别
		val gender_tag_udf: UserDefinedFunction = udf(
			(total: Long, maleTotal: Long, femaleTotal: Long) => {
				// 计算占比
				val maleRate: Double = maleTotal / total.toDouble
				val femaleRate: Double = femaleTotal / total.toDouble
				if(maleRate >= 0.6){ // usg = 男性
					rulesMapBroadcast.value("0")
				}else if(femaleRate >= 0.6){ // usg =女性
					rulesMapBroadcast.value("1")
				}else{ // usg = 中性
					rulesMapBroadcast.value("-1")
				}
			}
		)
		
		// 5.3 获取画像标签数据
		val modelDF: DataFrame = genderDF
			.select(
				$"userId", //
				gender_tag_udf($"total", $"maleTotal", $"femaleTotal").as("usg")
			)
		//modelDF.printSchema()
		//modelDF.show(100, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
	
	
	/**
	 * 针对数据集进行特征工程：特征提取、特征转换及特征选择
	 * @param dataframe 数据集
	 * @return 数据集，包含特征列features: Vector类型和标签列label
	 */
	def featuresTransform(dataframe: DataFrame): DataFrame = {
		/*
			// 1. 提取特征，封装值features向量中
			// 2. 将标签label索引化
			// 3. 将类别特征索引化
		 */
		// a. 特征向量化
		val assembler: VectorAssembler = new VectorAssembler()
			.setInputCols(Array("color", "product"))
			.setOutputCol("raw_features")
		val df1: DataFrame = assembler.transform(dataframe)
		
		// b. 类别特征进行索引
		val vectorIndexer: VectorIndexerModel = new VectorIndexer()
			.setInputCol("raw_features")
			.setOutputCol("features")
			.setMaxCategories(30)
			.fit(df1)
		val df2: DataFrame = vectorIndexer.transform(df1)
		
		// c. 返回特征数据
		df2
	}
	
	/**
	 * 使用决策树分类算法训练模型，返回DecisionTreeClassificationModel模型
	 * @return
	 */
	def trainModel(dataframe: DataFrame): DecisionTreeClassificationModel = {
		// a. 数据划分为训练数据集和测试数据集
		val Array(trainingDF, testingDF) = dataframe.randomSplit(Array(0.8, 0.2), seed = 123L)
		// b. 构建决策树分类器
		val dtc: DecisionTreeClassifier = new DecisionTreeClassifier()
			.setFeaturesCol("features")
			.setLabelCol("label")
			.setPredictionCol("prediction")
			.setImpurity("gini") // 基尼系数
			.setMaxDepth(5) // 树的深度
			.setMaxBins(32) // 树的叶子数目
		// c. 训练模型
		logWarning("正在训练模型...................................")
		val dtcModel: DecisionTreeClassificationModel = dtc.fit(trainingDF)
		// d. 模型评估
		val predictionDF: DataFrame = dtcModel.transform(testingDF)
		println(s"accuracy = ${modelEvaluate(predictionDF, "accuracy")}")
		// e. 返回模型
		dtcModel
	}
	
	/**
	 * 模型评估，返回计算分类指标值
	 * @param dataframe 预测结果的数据集
	 * @param metricName 分类评估指标名称，支持：f1、weightedPrecision、weightedRecall、accuracy
	 * @return
	 */
	def modelEvaluate(dataframe: DataFrame, metricName: String): Double = {
		// a. 构建多分类分类器
		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("label")
			.setPredictionCol("prediction")
			// 指标名称，
			.setMetricName(metricName)
		// b. 计算评估指标
		val metric: Double = evaluator.evaluate(dataframe)
		// c. 返回指标
		metric
	}
	
	/**
	 * 使用决策树分类算法训练模型，返回PipelineModel模型
	 * @return
	 */
	def trainPipelineModel(dataframe: DataFrame): PipelineModel = {
		// 数据划分为训练数据集和测试数据集
		val Array(trainingDF, testingDF) = dataframe.randomSplit(Array(0.8, 0.2), seed = 123)
		
		// a. 特征向量化
		val assembler: VectorAssembler = new VectorAssembler()
			.setInputCols(Array("color", "product"))
			.setOutputCol("raw_features")
		
		// b. 类别特征进行索引
		val vectorIndexer: VectorIndexer = new VectorIndexer()
			.setInputCol("raw_features")
			.setOutputCol("features")
			.setMaxCategories(30)
		
		// c. 构建决策树分类器
		val dtc: DecisionTreeClassifier = new DecisionTreeClassifier()
			.setFeaturesCol("features")
			.setLabelCol("label")
			.setPredictionCol("prediction")
			.setImpurity("gini") // 基尼系数
			.setMaxDepth(5) // 树的深度
			.setMaxBins(32) // 树的叶子数目
		
		// TODO: 构建Pipeline管道对象，组合模型学习器（算法）和转换器（模型）
		val pipeline: Pipeline = new Pipeline()
    		.setStages(Array(assembler, vectorIndexer, dtc))
		// 训练模型，使用训练数据集
		val pipelineModel: PipelineModel = pipeline.fit(trainingDF)
		
		// f. 模型评估
		val predictionDF: DataFrame = pipelineModel.transform(testingDF)
		predictionDF.show(100, truncate = false)
		println(s"accuracy = ${modelEvaluate(predictionDF, "accuracy")}")
		
		// 返回模型
		pipelineModel
	}
	
	/**
	 * 调整算法超参数，找出最优模型
	 * @param dataframe 数据集
	 * @return
	 */
	def trainBestModel(dataframe: DataFrame): PipelineModel = {
		// a. 特征向量化
		val assembler: VectorAssembler = new VectorAssembler()
			.setInputCols(Array("color", "product"))
			.setOutputCol("raw_features")
		
		// b. 类别特征进行索引
		val vectorIndexer: VectorIndexer = new VectorIndexer()
			.setInputCol("raw_features")
			.setOutputCol("features")
			.setMaxCategories(30)
		
		// c. 构建决策树分类器
		val dtc: DecisionTreeClassifier = new DecisionTreeClassifier()
			.setFeaturesCol("features")
			.setLabelCol("label")
			.setPredictionCol("prediction")
			.setImpurity("gini") // 基尼系数
			.setMaxDepth(5) // 树的深度
			.setMaxBins(32) // 树的叶子数目
		
		// 构建Pipeline管道对象，组合模型学习器（算法）和转换器（模型）
		val pipeline: Pipeline = new Pipeline()
			.setStages(Array(assembler, vectorIndexer, dtc))
		
		// TODO: 创建网格参数对象实例，设置算法中超参数的值
		val paramGrid: Array[ParamMap] = new ParamGridBuilder()
			.addGrid(dtc.impurity, Array("gini", "entropy"))
			.addGrid(dtc.maxDepth, Array(5, 10))
			.addGrid(dtc.maxBins, Array(32, 64))
			.build()
		
		// f. 多分类评估器
		val evaluator = new MulticlassClassificationEvaluator()
			.setLabelCol("label")
			.setPredictionCol("prediction")
			// 指标名称，支持：f1、weightedPrecision、weightedRecall、accuracy
			.setMetricName("accuracy")
		
		// TODO: 创建训练验证分割实例对象，设置算法、评估器和数据集占比
		val trainValidationSplit = new TrainValidationSplit()
			.setEstimator(pipeline) // 算法，应用数据训练获取模型
			.setEvaluator(evaluator) // 评估器，对训练模型进行评估
			.setEstimatorParamMaps(paramGrid) // 算法超参数，自动组合算法超参数，训练模型并评估，获取最佳模型
			// 80% of the data will be used for training and the remaining 20% for validation.
			.setTrainRatio(0.8)
		// 传递数据集，训练模型
		val splitModel: TrainValidationSplitModel = trainValidationSplit.fit(dataframe)
		
		// TODO： 获取最佳模型
		val pipelineModel: PipelineModel = splitModel.bestModel.asInstanceOf[PipelineModel]
		
		// 返回获取最佳模型
		pipelineModel
	}
	
}

object UsgTagModel {
	def main(args: Array[String]): Unit = {
		val tagModel = new UsgTagModel()
		tagModel.executeModel(378L)
	}
}