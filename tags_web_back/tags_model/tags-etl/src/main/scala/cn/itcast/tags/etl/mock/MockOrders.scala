package cn.itcast.tags.etl.mock

import java.util.Properties

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import scala.util.Random

object MockOrders {

	def main(args: Array[String]): Unit = {

		// 1. 构建SparkSession实例对象
		val spark: SparkSession = SparkSession.builder()
	    	.appName(this.getClass.getSimpleName.stripSuffix("$"))
	    	.master("local[4]")
	    	.config("spark.sql.shuffle.partitions", 4)
	    	.getOrCreate()
		import org.apache.spark.sql.functions._
		import spark.implicits._
		
		// 1. 直接从MySQL数据库表读取数据
		val ordersDF: DataFrame = spark.read
			.jdbc(
				"jdbc:mysql://bigdata-cdh01.itcast.cn:3306/tags_dat?" + //
					"user=root&password=123456&driver=com.mysql.jdbc.Driver", //
				"tbl_orders",
				new Properties() //
			)
		
		// 2. 自定义UDF函数，处理UserId: 订单表数据中会员ID -> memberid
		val user_id_udf: UserDefinedFunction = udf(
			(userId: String) => {
				if(userId.toInt >= 950){
					val id = new Random().nextInt(950) + 1
					id.toString
				}else{
					userId
				}
			}
		)
		
		// 3. 自定义UDF函数，处理paymentcode
		val paycodeList = List("alipay", "wxpay", "chinapay", "cod")
		val pay_code_udf: UserDefinedFunction = udf(
			(paymentcode: String) => {
				if(!paycodeList.contains(paymentcode)){
					val index: Int = new Random().nextInt(4)
					paycodeList(index)
				}else{
					paymentcode
				}
			}
		)
		
		// 4. 自定义UDF函数，处理paymentName
		val payMap: Map[String, String] = Map(
			"alipay" -> "支付宝", "wxpay" -> "微信支付", //
			"chinapay" -> "银联支付", "cod" -> "货到付款"
		)
		val pay_name_udf = udf(
			(paymentcode: String) => {
				payMap(paymentcode)
			}
		)
		
		
		// 5. 将会员ID值和支付方式值，使用UDF函数
		val newOrdersDF: DataFrame = ordersDF
			.withColumn("memberId", user_id_udf($"memberId"))
			.withColumn("paymentCode", pay_code_udf($"paymentCode"))
			.withColumn("paymentName", pay_name_udf($"paymentCode"))
			// 修改订单时间
			.withColumn(
				"finishTime", //
				unix_timestamp(
					date_add(from_unixtime($"finishTime"), 350), //
					"yyyy-MM-dd"
				)
			)

		// 6. 保存订单数据到MySQL表中
		newOrdersDF
            .write
            .mode(SaveMode.Append)
            .jdbc(
			    "jdbc:mysql://bigdata-cdh01.itcast.cn:3306/tags_dat?" + //
				    "user=root&password=123456&driver=com.mysql.jdbc.Driver", //
			    "tbl_tag_orders",
			    new Properties() //
		    )

		// 应用结束，关闭资源
		spark.stop()
	}

}
