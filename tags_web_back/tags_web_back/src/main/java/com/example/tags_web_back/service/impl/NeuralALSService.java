package com.example.tags_web_back.service;

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;

@Service
public class NeuralALSService {

    // 注入用户数据访问层接口，用于操作数据库
    @Autowired
    private UserMapper userMapper;

    // 定义一个方法，用于根据参数查询用户和商品的数据
    public List<User> getUserData(int page, int size) {
        // 计算查询的起始和结束位置
        int start = (page - 1) * size;
        int end = start + size;
        // 调用用户数据访问层接口的方法，根据起始和结束位置查询用户和商品的数据
        return userMapper.getUserData(start, end);
    }
}