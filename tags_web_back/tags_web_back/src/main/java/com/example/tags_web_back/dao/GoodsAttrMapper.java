package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.Tags;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;


@Mapper
public interface GoodsAttrMapper {

    @Select("SELECT pid, name, id, pre FROM goods_data ORDER BY pre DESC LIMIT #{start}, #{end}")
    List<Attr> getGoodsAttrData(@Param("start") int start, @Param("end") int end);
}