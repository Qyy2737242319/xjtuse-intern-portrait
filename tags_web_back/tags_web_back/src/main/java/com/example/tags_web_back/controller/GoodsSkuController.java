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
@RequestMapping("/api/product/sku")
public class GoodsSkuController {

    // 注入sku服务
    @Autowired
    private SkuService skuService;

    // 获取sku列表
    @GetMapping("/list")
    public SkuResponseData getSkuList(
            @RequestParam("pageNo") int pageNo,
            @RequestParam("pageSize") int pageSize
    ) {
        // 调用sku服务的方法，传入分页参数
        return skuService.getSkuList(pageNo, pageSize);
    }

    // 获取sku详情
    @GetMapping("/info")
    public SkuInfoData getSkuInfo(@RequestParam("skuId") int skuId) {
        // 调用sku服务的方法，传入sku id
        return skuService.getSkuInfo(skuId);
    }

    // 上架sku
    @PostMapping("/sale")
    public SkuResponseData saleSku(@RequestParam("skuId") int skuId) {
        // 调用sku服务的方法，传入sku id
        return skuService.saleSku(skuId);
    }

    // 下架sku
    @PostMapping("/cancel")
    public SkuResponseData cancelSale(@RequestParam("skuId") int skuId) {
        // 调用sku服务的方法，传入sku id
        return skuService.cancelSale(skuId);
    }

    // 删除sku
    @DeleteMapping("/remove")
    public SkuResponseData removeSku(@RequestParam("skuId") int skuId) {
        // 调用sku服务的方法，传入sku id
        return skuService.removeSku(skuId);
    }
}