package cn.itcast.tags.meta

import org.apache.spark.sql.Column

/**
 * 从HDFS文件系统读取数据，文件格式为csv类型，首行为列名称
	inType=hdfs
	inPath=/apps/datas/tbl_logs
	sperator=\t
	selectFieldNames=global_user_id,loc_url,log_time
 */
case class HdfsMeta(
	                   inPath: String,
	                   sperator: String,
	                   selectFieldNames: Array[Column]
                   )

object HdfsMeta{
	
	/**
	 * 将Map集合数据解析到HdfsMeta中
	 * @param ruleMap map集合
	 * @return
	 */
	def getHdfsMeta(ruleMap: Map[String, String]): HdfsMeta = {
		
		// 将选择字段构建为Column对象
		import org.apache.spark.sql.functions.col
		val fieldColumns: Array[Column] = ruleMap("selectFieldNames")
			.split(",")
			.map{field => col(field)}
		
		// 创建HdfsMeta对象并返回
		HdfsMeta(
			ruleMap("inPath"), //
			ruleMap("sperator"), //
			fieldColumns
		)
	}
}