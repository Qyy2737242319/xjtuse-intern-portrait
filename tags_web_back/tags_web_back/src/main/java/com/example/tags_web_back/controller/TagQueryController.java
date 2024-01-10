package com.example.tags_web_back.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import com.example.tags_web_back.model.User;

import java.util.ArrayList;

@RestController
public class TagQueryController {
    @RequestMapping("/hello")
    @ResponseBody
    public ArrayList<User> hello(@RequestParam(name = "tagid") int tagid) {
        ArrayList<User> user = new ArrayList<User>();

        return user;
    }
}
