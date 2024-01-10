package com.example.tags_web_back.service;

import com.example.tags_web_back.model.User;

import java.util.ArrayList;
import java.util.Optional;


public interface TagQueryService {
    Optional<ArrayList<User>> getUser(int tagid);
}
