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
public class GoodsAttrService {

    // 注入attr仓库
    @Autowired
    private AttrRepository attrRepository;

    // 获取attr列表
    public AttrResponseData getAttrList(int c1Id, int c2Id, int c3Id) {
        // 查询attr仓库，根据分类id
        List<Attr> list = attrRepository.findByCategoryId(c1Id, c2Id, c3Id);
        // 封装响应数据
        AttrResponseData data = new AttrResponseData();
        data.setData(list);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 添加或更新attr
    public AttrResponseData addOrUpdateAttr(Attr attr) {
        // 保存或更新attr仓库，根据attr对象
        attrRepository.saveOrUpdate(attr);
        // 封装响应数据
        AttrResponseData data = new AttrResponseData();
        data.setCode(200);
        data.setMessage(attr.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 删除attr
    public AttrResponseData removeAttr(int attrId) {
        // 删除attr仓库，根据attr id
        attrRepository.deleteById(attrId);
        // 封装响应数据
        AttrResponseData data = new AttrResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }
}