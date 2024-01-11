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
public class GoodsSkuService {

    // 注入sku仓库
    @Autowired
    private SkuRepository skuRepository;

    // 获取sku列表
    public SkuResponseData getSkuList(int pageNo, int pageSize) {
        // 分页查询sku仓库
        Page<SkuData> page = skuRepository.findAll(pageNo, pageSize);
        // 封装响应数据
        SkuResponseData data = new SkuResponseData();
        data.setRecords(page.getContent());
        data.setTotal(page.getTotalElements());
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 获取sku详情
    public SkuInfoData getSkuInfo(int skuId) {
        // 查询sku仓库，根据sku id
        SkuData sku = skuRepository.findById(skuId);
        // 封装响应数据
        SkuInfoData data = new SkuInfoData();
        data.setData(sku);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 上架sku
    public SkuResponseData saleSku(int skuId) {
        // 更新sku仓库，根据sku id，将isSale设为1
        skuRepository.updateIsSale(skuId, 1);
        // 封装响应数据
        SkuResponseData data = new SkuResponseData();
        data.setCode(200);
        data.setMessage("上架成功");
        return data;
    }

    // 下架sku
    public SkuResponseData cancelSale(int skuId) {
        // 更新sku仓库，根据sku id，将isSale设为0
        skuRepository.updateIsSale(skuId, 0);
        // 封装响应数据
        SkuResponseData data = new SkuResponseData();
        data.setCode(200);
        data.setMessage("下架成功");
        return data;
    }

    // 删除sku
    public SkuResponseData removeSku(int skuId) {
        // 删除sku仓库，根据sku id
        skuRepository.deleteById(skuId);
        // 封装响应数据
        SkuResponseData data = new SkuResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }
}