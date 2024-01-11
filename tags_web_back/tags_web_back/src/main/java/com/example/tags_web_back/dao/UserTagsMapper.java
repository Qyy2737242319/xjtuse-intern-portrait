package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.User_tags;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;
import java.util.Optional;

@Mapper
public interface UserTagsMapper {
    @Select("select userid from user_tags where tagsid = #{tagid}")
    ArrayList<Long> getuser(@Param("tagid") long tagid);
}
