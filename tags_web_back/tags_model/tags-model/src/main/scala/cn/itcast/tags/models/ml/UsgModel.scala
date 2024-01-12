package cn.itcast.tags.models.ml

import cn.itcast.tags.models.{AbstractModel, ModelType}
import cn.itcast.tags.tools.TagTools
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
 * 挖掘类型标签模型开发：用户购物性别标签模型
 */
class UsgModel extends AbstractModel("用户购物性别USG", ModelType.ML){
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
		val predictionDF: DataFrame = goodsDF.select($"userId", $"label".as("prediction"))
		
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
	
	
}

object UsgModel {
	def main(args: Array[String]): Unit = {
		val tagModel = new UsgModel()
		tagModel.executeModel(378L)
	}
}