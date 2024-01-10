package com.example.tags_web_back.dao;

import com.example.tags_web_back.model.User;
import org.apache.ibatis.annotations.*;

import java.util.ArrayList;
import java.util.Optional;

@Mapper
public interface UserMapper {
    @Select({
            "<script>",
            "SELECT id, username, email FROM tbl_users WHERE id IN",
            "<foreach item='userid' collection='userid' open='[' separator=',' close=']'>",
            "#{userid}",
            "</foreach>",
            "</script>"
    })
    Optional<ArrayList<User>> getUsers(Optional<ArrayList<Integer>> userid);

}
