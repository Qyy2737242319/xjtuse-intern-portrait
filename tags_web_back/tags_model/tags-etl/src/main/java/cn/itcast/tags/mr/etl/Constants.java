package cn.itcast.tags.mr.etl;

import org.apache.hadoop.hbase.util.Bytes;

import java.util.ArrayList;
import java.util.List;

/**
 * 定义常量
 */
interface Constants {
	// hive表数据目录
	String INPUT_PATH = "hdfs://bigdata-cdh01.itcast.cn:8020/user/hive/warehouse/tags_dat.db/tbl_logs";
	// 生成的hfile目录
	String HFILE_PATH = "hdfs://bigdata-cdh01.itcast.cn:8020/datas/output_hfile/tbl_logs";
	// 表名
	String TABLE_NAME = "tbl_logs";
	// 列簇名称
	byte[] COLUMN_FAMILY = Bytes.toBytes("detail");

	// 表字段
	List<byte[]> list = new ArrayList<byte[]>() {
		private static final long serialVersionUID = -6125158551837044300L;
		{ //
			add(Bytes.toBytes("id"));
			add(Bytes.toBytes("log_id"));
			add(Bytes.toBytes("remote_ip"));
			add(Bytes.toBytes("site_global_ticket"));
			add(Bytes.toBytes("site_global_session"));
			add(Bytes.toBytes("global_user_id"));
			add(Bytes.toBytes("cookie_text"));
			add(Bytes.toBytes("user_agent"));
			add(Bytes.toBytes("ref_url"));
			add(Bytes.toBytes("loc_url"));
			add(Bytes.toBytes("log_time"));
		} //
	};

}