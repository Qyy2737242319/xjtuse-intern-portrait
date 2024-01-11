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
public class AclPermissionService {

    // 注入permission仓库
    @Autowired
    private PermissionRepository permissionRepository;

    // 获取permission列表
    public PermissionResponseData getPermissionList() {
        // 查询permission仓库
        List<Permission> list = permissionRepository.findAll();
        // 封装响应数据
        PermissionResponseData data = new PermissionResponseData();
        data.setData(list);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 添加或更新permission
    public PermissionResponseData addOrUpdatePermission(Permission permission) {
        // 保存或更新permission仓库，根据permission对象
        permissionRepository.saveOrUpdate(permission);
        // 封装响应数据
        PermissionResponseData data = new PermissionResponseData();
        data.setCode(200);
        data.setMessage(permission.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 删除permission
    public PermissionResponseData removePermission(int permissionId) {
        // 删除permission仓库，根据permission id
        permissionRepository.deleteById(permissionId);
        // 封装响应数据
        PermissionResponseData data = new PermissionResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }
}