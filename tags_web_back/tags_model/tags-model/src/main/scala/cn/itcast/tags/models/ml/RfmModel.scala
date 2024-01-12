package cn.itcast.tags.models.ml

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, DecimalType}
import org.apache.spark.storage.StorageLevel

/**
 * 挖掘类型标签模型开发：客户价值模型RFM
 */
class RfmModel extends AbstractModel("客户价值RFM", ModelType.ML){
	/*
	361	客户价值
		362	高价值		0
		363	中上价值		1
		364	中价值		2
		365	中下价值		3
		366	超低价值		4
	 */
	override def doTag(businessDF: DataFrame, tagDF: DataFrame): DataFrame = {
		val session: SparkSession = businessDF.sparkSession
		import session.implicits._
		
		/*
		root
		 |-- memberid: string (nullable = true)
		 |-- ordersn: string (nullable = true)
		 |-- orderamount: string (nullable = true)
		 |-- finishtime: string (nullable = true)
		 */
		//businessDF.printSchema()
		//businessDF.show(10, truncate = false)
		
		/*
		root
		 |-- id: long (nullable = false)
		 |-- name: string (nullable = true)
		 |-- rule: string (nullable = true)
		 |-- level: integer (nullable = true)
		 */
		//tagDF.printSchema()
		/*
		|id |name|rule|level|
		+---+----+----+-----+
		|362|高价值 |0   |5    |
		|363|中上价值|1   |5    |
		|364|中价值 |2   |5    |
		|365|中下价值|3   |5    |
		|366|超低价值|4   |5    |
		+---+----+----+-----+
		 */
		//tagDF.filter($"level" === 5).show(10, truncate = false)
		
		
		/*
		TODO: 1、计算每个用户RFM值
			按照用户memberid分组，然后进行聚合函数聚合统计
			R：消费周期，finishtime
				日期时间函数：current_timestamp、from_unixtimestamp、datediff
			F: 消费次数 ordersn
				count
			M：消费金额 orderamount
				sum
		 */
		val rfmDF: DataFrame = businessDF
			// a. 按照memberid分组，对每个用户的订单数据句话操作
    		.groupBy($"memberid")
    		.agg(
			    max($"finishtime").as("max_finishtime"), //
			    count($"ordersn").as("frequency"), //
			    sum(
				    $"orderamount".cast(DataTypes.createDecimalType(10, 2))
			    ).as("monetary") //
		    )
			// 计算R值
    		.select(
			    $"memberid".as("userId"), //
			    // 计算R值：消费周期
			    datediff(
				    current_timestamp(), from_unixtime($"max_finishtime")
			    ).as("recency"), //
			    $"frequency", //
			    $"monetary"
		    )
		//rfmDF.printSchema()
		//rfmDF.show(10, truncate = false)
		
		/*
		TODO: 2、按照规则给RFM进行打分（RFM_SCORE)
			R: 1-3天=5分，4-6天=4分，7-9天=3分，10-15天=2分，大于16天=1分
	        F: ≥200=5分，150-199=4分，100-149=3分，50-99=2分，1-49=1分
	        M: ≥20w=5分，10-19w=4分，5-9w=3分，1-4w=2分，<1w=1分
			
			使用CASE WHEN ..  WHEN... ELSE .... END
		 */
		// R 打分条件表达式
		val rWhen = when(col("recency").between(1, 3), 5.0) //
			.when(col("recency").between(4, 6), 4.0) //
			.when(col("recency").between(7, 9), 3.0) //
			.when(col("recency").between(10, 15), 2.0) //
			.when(col("recency").geq(16), 1.0) //
		// F 打分条件表达式
		val fWhen = when(col("frequency").between(1, 49), 1.0) //
			.when(col("frequency").between(50, 99), 2.0) //
			.when(col("frequency").between(100, 149), 3.0) //
			.when(col("frequency").between(150, 199), 4.0) //
			.when(col("frequency").geq(200), 5.0) //
		// M 打分条件表达式
		val mWhen = when(col("monetary").lt(10000), 1.0) //
			.when(col("monetary").between(10000, 49999), 2.0) //
			.when(col("monetary").between(50000, 99999), 3.0) //
			.when(col("monetary").between(100000, 199999), 4.0) //
			.when(col("monetary").geq(200000), 5.0) //
		val rfmScoreDF: DataFrame = rfmDF.select(
			$"userId", //
			rWhen.as("r_score"), //
			fWhen.as("f_score"), //
			mWhen.as("m_score") //
		)
		//rfmScoreDF.printSchema()
		//rfmScoreDF.show(50, truncate = false)
		
		/*
		TODO: 3、使用RFM_SCORE进行聚类，对用户进行分组
			KMeans算法，其中K=5
		 */
		// 3.1 组合R\F\M列为特征值features
		val assembler: VectorAssembler = new VectorAssembler()
			.setInputCols(Array("r_score", "f_score", "m_score"))
			.setOutputCol("features")
		val featuresDF: DataFrame = assembler.transform(rfmScoreDF)
		// 将训练数据缓存
		featuresDF.persist(StorageLevel.MEMORY_AND_DISK)
		
		// 3.2 使用KMeans算法聚类，训练模型
		/*
			val kMeansModel: KMeansModel = new KMeans()
				.setFeaturesCol("features")
				.setPredictionCol("prediction") // 由于K=5，所以预测值prediction范围：0,1,2,3,4
				// K值设置，类簇个数
				.setK(5)
				.setMaxIter(20)
				.setInitMode("k-means||")
				// 训练模型
				.fit(featuresDF)
			// WSSSE = 0.9977375565642177
			println(s"WSSSE = ${kMeansModel.computeCost(featuresDF)}")
		*/
		val kMeansModel: KMeansModel = trainModel(featuresDF)
		
		// 3.3. 使用模型预测
		val predictionDF: DataFrame = kMeansModel.transform(featuresDF)
		/*
		root
		 |-- userId: string (nullable = true)
		 |-- r_score: double (nullable = true)
		 |-- f_score: double (nullable = true)
		 |-- m_score: double (nullable = true)
		 |-- features: vector (nullable = true)
		 |-- prediction: integer (nullable = true)
		 */
		//predictionDF.printSchema()
		//predictionDF.show(50, truncate = false)
		
		// 3.4 获取类簇中心点
		val centerIndexArray: Array[((Int, Double), Int)] = kMeansModel
			.clusterCenters
			// 返回值类型：: Array[(linalg.Vector, Int)]
    		.zipWithIndex // (vector1, 0), (vector2, 1), ....
			// TODO: 对每个类簇向量进行累加和：R + F + M
			.map{case(clusterVector, clusterIndex) =>
				// rfm表示将R + F + M之和，越大表示客户价值越高
				val rfm: Double = clusterVector.toArray.sum
				clusterIndex -> rfm
			}
			// 按照rfm值进行降序排序
			.sortBy(tuple => - tuple._2)
			// 再次进行拉链操作
			.zipWithIndex
		//centerIndexArray.foreach(println)
		
		// TODO： 4. 打标签
		// 4.1 获取属性标签规则rule和名称tagName，放在Map集合中
		val rulesMap: Map[String, String] = TagTools.convertMap(tagDF)
		//rulesMap.foreach(println)
		
		// 4.2 聚类类簇关联属性标签数据rule，对应聚类类簇与标签tagName
		val indexTagMap: Map[Int, String] = centerIndexArray
			.map{case((centerIndex, _), index) =>
				val tagName = rulesMap(index.toString)
				(centerIndex, tagName)
			}
			.toMap
		//indexTagMap.foreach(println)
		
		// 4.3 使用KMeansModel预测值prediction打标签
		// a. 将索引标签Map集合 广播变量广播出去
		val indexTagMapBroadcast = session.sparkContext.broadcast(indexTagMap)
		// b. 自定义UDF函数，传递预测值prediction，返回标签名称tagName
		val index_to_tag: UserDefinedFunction = udf(
			(clusterIndex: Int) => indexTagMapBroadcast.value(clusterIndex)
		)
		// c. 打标签
		val modelDF: DataFrame = predictionDF.select(
			$"userId", // 用户ID
			index_to_tag($"prediction").as("rfm")
		)
		//modelDF.printSchema()
		//modelDF.show(100, truncate = false)
		
		// 返回画像标签数据
		modelDF
	}
	
	/**
	 * 使用KMeans算法训练模型
	 * @param dataframe 数据集
	 * @return KMeansModel模型
	 */
	def trainModel(dataframe: DataFrame): KMeansModel = {
		// 使用KMeans聚类算法模型训练
		val kMeansModel: KMeansModel = new KMeans()
			.setFeaturesCol("features")
			.setPredictionCol("prediction")
			.setK(5) // 设置列簇个数：5
			.setMaxIter(20) // 设置最大迭代次数
			.fit(dataframe)
		println(s"WSSSE = ${kMeansModel.computeCost(dataframe)}")
		
		// 返回
		kMeansModel
	}
}

object RfmModel{
	def main(args: Array[String]): Unit = {
		val tagModel = new RfmModel()
		tagModel.executeModel(361L)
	}
}