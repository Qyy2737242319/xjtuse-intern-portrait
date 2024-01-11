package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.Tags;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.ArrayList;

@Mapper
public interface MicroscopicPortraitsMapper {
    @Select("SELECT tags.tag_id, tags.tag_level, tags.tag_des, tags.tag_pid " +
            "FROM tags " +
            "JOIN user_tags ON tags.tag_id = user_tags.tagsid " +
            "WHERE user_tags.userid = #{userid}")
    ArrayList<Tags> getJoinedTagsByUserId(@Param("userid") long userid);
}
