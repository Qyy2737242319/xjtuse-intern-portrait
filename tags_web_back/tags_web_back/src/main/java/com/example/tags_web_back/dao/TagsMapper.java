package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.Tags;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;

@Mapper
public interface TagsMapper {
    @Select("SELECT * FROM tags WHERE tag_level <= #{tag_level}")
    ArrayList<Tags> getTagsUnderLevel(int tag_level);
}
