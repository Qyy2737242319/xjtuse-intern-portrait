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
@RequestMapping("/api/product/spu")
public class GoodsSpuController {

    // 注入spu服务
    @Autowired
    private SpuService spuService;

    // 获取spu列表
    @GetMapping("/list")
    public HasSpuResponseData getSpuList(
            @RequestParam("pageNo") int pageNo,
            @RequestParam("pageSize") int pageSize,
            @RequestParam("c3Id") int c3Id
    ) {
        // 调用spu服务的方法，传入分页参数和分类id
        return spuService.getSpuList(pageNo, pageSize, c3Id);
    }

    // 获取sku列表
    @GetMapping("/sku")
    public SkuInfoData getSkuList(@RequestParam("spuId") int spuId) {
        // 调用spu服务的方法，传入spu id
        return spuService.getSkuList(spuId);
    }

    // 添加或更新spu
    @PostMapping("/addOrUpdate")
    public SpuResponseData addOrUpdateSpu(@RequestBody SpuData spu) {
        // 调用spu服务的方法，传入spu对象
        return spuService.addOrUpdateSpu(spu);
    }

    // 添加或更新sku
    @PostMapping("/addOrUpdateSku")
    public SkuResponseData addOrUpdateSku(@RequestBody SkuData sku) {
        // 调用spu服务的方法，传入sku对象
        return spuService.addOrUpdateSku(sku);
    }

    // 删除spu
    @DeleteMapping("/remove")
    public SpuResponseData removeSpu(@RequestParam("spuId") int spuId) {
        // 调用spu服务的方法，传入spu id
        return spuService.removeSpu(spuId);
    }
}