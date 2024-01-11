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
@RequestMapping("/api/acl/menu")
public class AclPermissionController {

    // 注入菜单服务类
    @Autowired
    private MenuService menuService;

    // 获取所有的权限数据
    @GetMapping("/all")
    public PermissionResponseData getAllPermission() {
        PermissionList permissionList = menuService.getAllPermission();
        return new PermissionResponseData(200, "成功", permissionList);
    }

    // 添加或更新菜单
    @PostMapping("/saveOrUpdate")
    public ResponseData saveOrUpdateMenu(@RequestBody MenuParams menuParams) {
        menuService.saveOrUpdateMenu(menuParams);
        return new ResponseData(200, "成功");
    }

    // 删除菜单
    @DeleteMapping("/remove/{id}")
    public ResponseData removeMenu(@PathVariable("id") int id) {
        menuService.removeMenu(id);
        return new ResponseData(200, "成功");
    }
}