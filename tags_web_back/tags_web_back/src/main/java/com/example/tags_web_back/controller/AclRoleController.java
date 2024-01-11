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
@RequestMapping("/api/acl/role")
public class AclRoleController {

    // 注入角色服务类
    @Autowired
    private RoleService roleService;

    // 注入菜单服务类
    @Autowired
    private MenuService menuService;

    // 获取所有的角色数据
    @GetMapping("/all")
    public RoleResponseData getAllRole(@RequestParam("pageNo") int pageNo, @RequestParam("pageSize") int pageSize, @RequestParam("keyword") String keyword) {
        RoleList roleList = roleService.getAllRole(pageNo, pageSize, keyword);
        return new RoleResponseData(200, "成功", roleList);
    }

    // 添加或更新角色
    @PostMapping("/saveOrUpdate")
    public ResponseData saveOrUpdateRole(@RequestBody RoleData roleData) {
        roleService.saveOrUpdateRole(roleData);
        return new ResponseData(200, "成功");
    }

    // 删除角色
    @DeleteMapping("/remove/{id}")
    public ResponseData removeRole(@PathVariable("id") int id) {
        roleService.removeRole(id);
        return new ResponseData(200, "成功");
    }

    // 获取角色对应的菜单数据
    @GetMapping("/menu/{id}")
    public MenuResponseData getMenuByRole(@PathVariable("id") int id) {
        MenuList menuList = menuService.getMenuByRole(id);
        return new MenuResponseData(200, "成功", menuList);
    }

    // 设置角色对应的菜单权限
    @PostMapping("/permission/{id}")
    public ResponseData setPermissionByRole(@PathVariable("id") int id, @RequestBody int[] permissionId) {
        menuService.setPermissionByRole(id, permissionId);
        return new ResponseData(200, "成功");
    }
}