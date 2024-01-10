package com.example.tags_web_back.service.impl;

import com.example.tags_web_back.dao.UserMapper;
import com.example.tags_web_back.dao.UserTagsMapper;
import com.example.tags_web_back.model.User;
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
    public Optional<ArrayList<User>> getUser(int tagid) {
        Optional<ArrayList<Integer>> userid = userTagsMapper.getuser(tagid);
        Optional<ArrayList<User>> users = userMapper.getUsers(userid);
        return users;
    }
}
