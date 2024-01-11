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
public class AclUserService {

    // 注入user仓库
    @Autowired
    private UserRepository userRepository;

    // 获取user列表
    public UserResponseData getUserList(String keyword, int pageNo, int pageSize) {
        // 分页查询user仓库，根据关键字
        Page<User> page = userRepository.findByKeyword(keyword, pageNo, pageSize);
        // 封装响应数据
        UserResponseData data = new UserResponseData();
        data.setRecords(page.getContent());
        data.setTotal(page.getTotalElements());
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 添加或更新user
    public UserResponseData addOrUpdateUser(User user) {
        // 保存或更新user仓库，根据user对象
        userRepository.saveOrUpdate(user);
        // 封装响应数据
        UserResponseData data = new UserResponseData();
        data.setCode(200);
        data.setMessage(user.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 删除user
    public UserResponseData removeUser(int userId) {
        // 删除user仓库，根据user id
        userRepository.deleteById(userId);
        // 封装响应数据
        UserResponseData data = new UserResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }

    // 获取所有角色列表
    public AllRoleResponseData getAllRole(int userId) {
        // 查询角色仓库，根据user id
        List<Role> list = roleRepository.findByUserId(userId);
        // 封装响应数据
        AllRoleResponseData data = new AllRoleResponseData();
        data.setData(list);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 设置角色
    public UserResponseData setRole(int userId, List<Integer> roleIdList) {
        // 更新角色仓库，根据user id和角色id列表
        roleRepository.updateByUserIdAndRoleIdList(userId, roleIdList);
        // 封装响应数据
        UserResponseData data = new UserResponseData();
        data.setCode(200);
        data.setMessage("分配角色成功");
        return data;
    }
}