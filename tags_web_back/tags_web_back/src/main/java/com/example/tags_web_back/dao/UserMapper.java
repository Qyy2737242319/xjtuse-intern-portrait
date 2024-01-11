package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.User;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;
import java.util.Optional;

@Mapper
public interface UserMapper {
    @Select({
            "SELECT id, username, email FROM tbl_users WHERE id IN (${ids})"
    })
    ArrayList<Long> getUsers(@Param("ids") String ids);
}
