package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.Tags;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;

// 定义一个用户数据访问层接口，用于操作数据库
@Mapper
public interface NeuralALSMapper {

    // 定义一个方法，用于根据起始和结束位置查询用户和商品的数据
    // 使用 @Select 注解来编写 SQL 语句，根据预测用户满意度降序排序
    @Select("SELECT pid, name, id, pre FROM user_data ORDER BY pre DESC LIMIT #{start}, #{end}")
    List<User> getUserData(@Param("start") int start, @Param("end") int end);
}
