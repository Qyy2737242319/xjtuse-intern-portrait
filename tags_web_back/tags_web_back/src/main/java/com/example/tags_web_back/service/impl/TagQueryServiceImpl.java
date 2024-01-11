package com.example.tags_web_back.service.impl;

import com.example.tags_web_back.dao.UserMapper;
import com.example.tags_web_back.dao.UserTagsMapper;
import com.example.tags_web_back.model.User;
import com.example.tags_web_back.model.User_tags;
import com.example.tags_web_back.service.TagQueryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Optional;
@Service
public class TagQueryServiceImpl implements TagQueryService {

    @Autowired
    private UserTagsMapper userTagsMapper;
    @Autowired
    private UserMapper userMapper;

    @Override
    public ArrayList<Long> getUser(int tagid) {
        ArrayList<Long> userid = userTagsMapper.getuser(tagid);
//        String ids = String.join(",", userid.toString());
//        System.out.println(ids);
//        ArrayList<User> users = userMapper.getUsers(ids);
        return userid;
    }
}
