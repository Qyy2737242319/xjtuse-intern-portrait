package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.Tags;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;

// 定义一个用户数据访问层接口，用于操作数据库
@Mapper
public interface AclPermissionMapper {

    // 定义一个方法，用于根据起始和结束位置查询用户id和权限
    // 使用 @Select 注解来编写 SQL 语句
    @Select("SELECT pid, name, id, pre FROM user_permission_data ORDER BY pre DESC LIMIT #{start}, #{end}")
    List<Permission> getPermissionData(@Param("start") int start, @Param("end") int end);
}