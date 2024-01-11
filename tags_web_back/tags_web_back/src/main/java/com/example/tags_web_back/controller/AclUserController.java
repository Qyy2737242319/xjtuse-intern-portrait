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
@RequestMapping("/api/acl/user")
public class AclUserController {

    // 注入用户服务
    @Autowired
    private UserService userService;

    // 注入RestTemplate
    @Autowired
    private RestTemplate restTemplate;

    // 获取用户信息
    @GetMapping("/info")
    public UserResponseData getUserInfo(
            @RequestParam("pageNo") int pageNo,
            @RequestParam("pageSize") int pageSize,
            @RequestParam("keyword") String keyword
    ) {
        // 调用用户服务的方法，传入参数
        return userService.getUserInfo(pageNo, pageSize, keyword);
    }

    // 添加或更新用户
    @PostMapping("/addOrUpdate")
    public UserResponseData addOrUpdateUser(@RequestBody User user) {
        // 调用用户服务的方法，传入用户对象
        return userService.addOrUpdateUser(user);
    }

    // 获取所有角色
    @GetMapping("/allRole")
    public AllRoleResponseData getAllRole(@RequestParam("userId") int userId) {
        // 调用用户服务的方法，传入用户id
        return userService.getAllRole(userId);
    }

    // 设置用户角色
    @PostMapping("/setRole")
    public UserResponseData setUserRole(@RequestBody SetRoleData data) {
        // 调用用户服务的方法，传入设置角色的数据对象
        return userService.setUserRole(data);
    }

    // 删除用户
    @DeleteMapping("/remove")
    public UserResponseData removeUser(@RequestParam("userId") int userId) {
        // 调用用户服务的方法，传入用户id
        return userService.removeUser(userId);
    }

    // 选择用户
    @GetMapping("/select")
    public UserResponseData selectUser(@RequestParam("userId") int userId) {
        // 调用用户服务的方法，传入用户id
        return userService.selectUser(userId);
    }
}