package com.example.tags_web_back.controller;

import com.example.tags_web_back.config.ApiResponse;
import com.example.tags_web_back.model.User_tags;
import com.example.tags_web_back.service.TagQueryService;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.example.tags_web_back.model.User;

import java.util.ArrayList;
import java.util.Optional;

@RestController
public class TagQueryController {
    @Autowired
    private TagQueryService tagQueryService;


    @GetMapping ("api/tagquery/{tagid}")
    @ResponseBody
    public ApiResponse tagquery(@PathVariable int tagid) {
        ArrayList<Long> user = tagQueryService.getUser(tagid);
        return ApiResponse.ok(user);
    }
}

