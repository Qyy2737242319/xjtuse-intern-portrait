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
@RequestMapping("/api/attr")
public class GoodsAttrController {

    // 注入属性服务
    @Autowired
    private AttrService attrService;

    // 获取属性列表
    @GetMapping("/list")
    public AttrResponseData getAttrList(
            @RequestParam("c1Id") int c1Id,
            @RequestParam("c2Id") int c2Id,
            @RequestParam("c3Id") int c3Id
    ) {
        // 调用属性服务的方法，传入分类id
        return attrService.getAttrList(c1Id, c2Id, c3Id);
    }

    // 添加或更新属性
    @PostMapping("/addOrUpdate")
    public AttrResponseData addOrUpdateAttr(@RequestBody Attr attr) {
        // 调用属性服务的方法，传入属性对象
        return attrService.addOrUpdateAttr(attr);
    }

    // 删除属性
    @DeleteMapping("/remove")
    public AttrResponseData removeAttr(@RequestParam("attrId") int attrId) {
        // 调用属性服务的方法，传入属性id
        return attrService.removeAttr(attrId);
    }
}