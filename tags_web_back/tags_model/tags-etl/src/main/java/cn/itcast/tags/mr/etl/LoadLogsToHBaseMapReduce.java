package cn.itcast.tags.mr.etl;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.HFileOutputFormat2;
import org.apache.hadoop.hbase.mapreduce.LoadIncrementalHFiles;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;

/**
 * 将Hive表数据转换为HFile文件并移动HFile到HBase
 */
public class LoadLogsToHBaseMapReduce extends Configured implements Tool {

	// 连接HBase Connection对象
	private static Connection connection = null ;

	/**
	 * 定义Mapper类，读取CSV格式数据，转换为Put对象，存储HBase表
	 */
	static class LoadLogsToHBase extends Mapper<LongWritable, Text, ImmutableBytesWritable, Put> {
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			// 按照分隔符分割数据，分隔符为 逗号
			// 1	20190813030355b18919aa	58.210.214.242	d5bbefc1-b018-411e-a5c5-f4ae42040585	225B3CEC05A99F22DA910E9ACF16CABA	424	_jzqa:
			String[] split = value.toString().split("\\t");
			if (split.length == Constants.list.size()) {
				// 构建Put对象，将每行数据转换为Put
				Put put = new Put(Bytes.toBytes(split[0]));
				for (int i = 1; i < Constants.list.size(); i++) {
					put.addColumn(//
						Constants.COLUMN_FAMILY, //
						Constants.list.get(i), //
						Bytes.toBytes(split[i]) //
					);
				}
				// 将数据输出
				context.write(new ImmutableBytesWritable(put.getRow()), put);
			}
		}
	}

	@Override
	public int run(String[] args) throws Exception {
		// a. 获取配置信息对象
		Configuration configuration = super.getConf() ;

		// b. 构建Job对象Job
		Job job = Job.getInstance(configuration);
		job.setJobName(this.getClass().getSimpleName());
		job.setJarByClass(LoadLogsToHBaseMapReduce.class);

		// c. 设置Job
		FileInputFormat.addInputPath(job, new Path(Constants.INPUT_PATH));

		job.setMapperClass(LoadLogsToHBase.class);
		job.setMapOutputKeyClass(ImmutableBytesWritable.class);
		job.setMapOutputValueClass(Put.class);

		// TODO: 设置输出格式为HFileOutputFormat2
		job.setOutputFormatClass(HFileOutputFormat2.class);

		// TODO: 判断输出目录是否存在，如果存在就删除
		FileSystem hdfs = FileSystem.get(configuration) ;
		Path outputPath = new Path(Constants.HFILE_PATH) ;
		if(hdfs.exists(outputPath)){
			hdfs.delete(outputPath, true) ;
		}
		// d. 设置输出路径
		FileOutputFormat.setOutputPath(job, outputPath);

		// TODO：获取HBase Table，对HFileOutputFormat2进行设置
		Table table = connection.getTable(TableName.valueOf(Constants.TABLE_NAME));
		HFileOutputFormat2.configureIncrementalLoad( //
			job, //
			table, // 目的获取HBase表的Region个数，决定生成HFile文件的个数，一个Region中对对应一个文件
			connection.getRegionLocator(TableName.valueOf(Constants.TABLE_NAME)) //
		);

		// 提交运行Job，返回是否执行成功
		boolean isSuccess = job.waitForCompletion(true);

		return isSuccess ? 0 : 1;
	}

	public static void main(String[] args)  throws Exception{
		// 获取Configuration对象，读取配置信息
		Configuration configuration = HBaseConfiguration.create();
		// 获取HBase 连接Connection对象
		connection = ConnectionFactory.createConnection(configuration);

		// 运行MapReduce将数据文件转换为HFile文件
		int status = ToolRunner.run(configuration, new LoadLogsToHBaseMapReduce(), args);
		System.out.println("HFile文件生成完毕!~~~");

		// TODO：运行成功时，加载HFile文件数据到HBase表中
		if (0 == status) {
			// 获取HBase Table句柄
			Admin admin = connection.getAdmin();
			Table table = connection.getTable(TableName.valueOf(Constants.TABLE_NAME));
			// 加载数据到表中
			LoadIncrementalHFiles load = new LoadIncrementalHFiles(configuration);
			load.doBulkLoad(
				new Path(Constants.HFILE_PATH), //
				admin, //
				table, //
				connection.getRegionLocator(TableName.valueOf(Constants.TABLE_NAME)) //
			);
			System.out.println("HFile文件移动完毕!~~~");
		}
	}

}
