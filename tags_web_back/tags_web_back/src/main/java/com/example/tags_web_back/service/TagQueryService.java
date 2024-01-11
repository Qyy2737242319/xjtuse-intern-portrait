package com.example.tags_web_back.service;

import com.example.tags_web_back.model.User;
import com.example.tags_web_back.model.User_tags;

import java.util.ArrayList;
import java.util.Optional;


public interface TagQueryService {
//    ArrayList<Long> getUser(int tagid);
    ArrayList<User> getUser(int tagid);
}
