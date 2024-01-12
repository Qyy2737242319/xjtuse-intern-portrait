package cn.itcast.tags.meta

import org.apache.spark.sql.Column

/**
 * 从Hive表中加载数据，SparkSession创建时与Hive集成已配置
		inType=hive
		hiveTable=tags_dat.tbl_logs
		selectFieldNames=global_user_id,loc_url,log_time
		## 分区字段及数据范围
		whereCondition=log_time#day#30
 */
case class HiveMeta(
                   hiveTable: String,
                   selectFieldNames: Array[Column],
                   whereCondition: String
                   )

object HiveMeta{
	
	/**
	 * 将Map集合数据解析到HiveMeta中
	 * @param ruleMap map集合
	 * @return
	 */
	def getHiveMeta(ruleMap: Map[String, String]): HiveMeta = {
		// 此处省略依据分组字段值构建WHERE CAUSE 语句
		// val whereCondition = ...
		
		// 将选择字段构建为Column对象
		import org.apache.spark.sql.functions.col
		val fieldColumns: Array[Column] = ruleMap("selectFieldNames")
			.split(",")
			.map{field => col(field)}
		
		// 创建HiveMeta对象并返回
		HiveMeta(
			ruleMap("hiveTable"), //
			fieldColumns, //
			null
		)
	}
}