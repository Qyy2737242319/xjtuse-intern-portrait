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
public class GoodsSpuService {

    // 注入spu仓库
    @Autowired
    private SpuRepository spuRepository;

    // 注入sku仓库
    @Autowired
    private SkuRepository skuRepository;

    // 获取spu列表
    public HasSpuResponseData getSpuList(int pageNo, int pageSize, int c3Id) {
        // 分页查询spu仓库，根据分类id
        Page<SpuData> page = spuRepository.findByC3Id(pageNo, pageSize, c3Id);
        // 封装响应数据
        HasSpuResponseData data = new HasSpuResponseData();
        data.setRecords(page.getContent());
        data.setTotal(page.getTotalElements());
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 获取sku列表
    public SkuInfoData getSkuList(int spuId) {
        // 查询sku仓库，根据spu id
        List<SkuData> list = skuRepository.findBySpuId(spuId);
        // 封装响应数据
        SkuInfoData data = new SkuInfoData();
        data.setData(list);
        data.setCode(200);
        data.setMessage("success");
        return data;
    }

    // 添加或更新spu
    public SpuResponseData addOrUpdateSpu(SpuData spu) {
        // 保存或更新spu仓库，根据spu对象
        spuRepository.saveOrUpdate(spu);
        // 封装响应数据
        SpuResponseData data = new SpuResponseData();
        data.setCode(200);
        data.setMessage(spu.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 添加或更新sku
    public SkuResponseData addOrUpdateSku(SkuData sku) {
        // 保存或更新sku仓库，根据sku对象
        skuRepository.saveOrUpdate(sku);
        // 封装响应数据
        SkuResponseData data = new SkuResponseData();
        data.setCode(200);
        data.setMessage(sku.getId() == null ? "添加成功" : "更新成功");
        return data;
    }

    // 删除spu
    public SpuResponseData removeSpu(int spuId) {
        // 删除spu仓库，根据spu id
        spuRepository.deleteById(spuId);
        // 封装响应数据
        SpuResponseData data = new SpuResponseData();
        data.setCode(200);
        data.setMessage("删除成功");
        return data;
    }
}