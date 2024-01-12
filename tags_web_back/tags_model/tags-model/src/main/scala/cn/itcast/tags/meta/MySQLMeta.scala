package cn.itcast.tags.meta

/**
 * RDBMS 关系型数据库元数据解析存储，具体数据字段格式如下所示：
	inType=mysql
	driver=com.mysql.jdbc.Driver
	url=jdbc:mysql://bigdata-cdh01.itcast.cn:3306/?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC
	user=root
	password=123456
	sql=SELECT id, gender FROM tags_dat.tbl_users
 */
case class MySQLMeta(
	                    driver: String,
	                    url: String,
	                    user: String,
	                    password: String,
	                    sql: String
                    )

object MySQLMeta{
	
	/**
	 * 将Map集合数据解析到RdbmsMeta中
	 * @param ruleMap map集合
	 * @return
	 */
	def getMySQLMeta(ruleMap: Map[String, String]): MySQLMeta = {
		// 获取SQL语句，赋以别名
		val sqlStr: String = s"( ${ruleMap("sql")} ) AS tmp"
		// 构建RdbmsMeta对象
		MySQLMeta(
			ruleMap("driver"),
			ruleMap("url"),
			ruleMap("user"),
			ruleMap("password"),
			sqlStr
		)
	}
	
}