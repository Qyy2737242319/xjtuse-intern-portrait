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
public class AclRoleService {

    // 注入role仓库
    @Autowired
    private RoleRepository roleRepository;

    // 获取role列表
    public RoleResponseData getRoleList(String keyword, int pageNo, int pageSize) {
        // 分页查询role仓库，根据关键字
        Page<Role> page = roleRepository.findByKeyword(keyword, pageNo, pageSize);
        // 封装响应数据
        RoleResponseData data = new RoleResponseData();
        data.setRecords(page.getContent());
        data.setTotal(page.getTotalElements());
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 添加或更新role
    public RoleResponseData addOrUpdateRole(Role role) {
        // 保存或更新role仓库，根据role对象
        roleRepository.saveOrUpdate(role);
        // 封装响应数据
        RoleResponseData data = new RoleResponseData();
        data.setCode(200);
        data.setMessage(role.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 删除role
    public RoleResponseData removeRole(int roleId) {
        // 删除role仓库，根据role id
        roleRepository.deleteById(roleId);
        // 封装响应数据
        RoleResponseData data = new RoleResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }

    // 获取所有菜单列表
    public MenuResponseData getAllMenuList(int roleId) {
        // 查询菜单仓库，根据role id
        List<Menu> list = menuRepository.findByRoleId(roleId);
        // 封装响应数据
        MenuResponseData data = new MenuResponseData();
        data.setData(list);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 设置权限
    public RoleResponseData setPermission(int roleId, List<Integer> permissionId) {
        // 更新权限仓库，根据role id和权限id列表
        permissionRepository.updateByRoleIdAndPermissionId(roleId, permissionId);
        // 封装响应数据
        RoleResponseData data = new RoleResponseData();
        data.setCode(200);
        data.setMessage("分配权限成功");
        return data;
    }
}