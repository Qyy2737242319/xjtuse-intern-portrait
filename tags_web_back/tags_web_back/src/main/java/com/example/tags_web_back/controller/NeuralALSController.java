package com.example.tags_web_back.controller;

import com.example.tags_web_back.config.ApiResponse;
import com.example.tags_web_back.dto.MicroscopicPortraitsRequest;
import com.example.tags_web_back.service.MicroscopicPortraitsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;

@RestController
public class NeuralALSController {
    // 注入用户服务类，用于调用业务逻辑和数据访问层
    @Autowired
    private UserService userService;

    // 定义一个方法，用于接收前端传递的参数
    @GetMapping
    public List<User> getData(@RequestParam("user_id") int user_id, @RequestParam("item_id") int item_id) {
        // 调用用户服务类的方法，根据参数查询用户和商品的数据，并返回给前端
        return userService.getUserData(user_id, item_id);
    }
}