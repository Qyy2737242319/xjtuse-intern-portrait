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
public class MicroscopicPortraitsController {
    final MicroscopicPortraitsService microscopicPortraitsService;

    public MicroscopicPortraitsController(@Autowired MicroscopicPortraitsService microscopicPortraitsService) {
        this.microscopicPortraitsService = microscopicPortraitsService;
    }

    @PostMapping("/api/microscopicPortraits")
    public ApiResponse microscopicPortraits(@RequestBody MicroscopicPortraitsRequest microscopicPortraitsRequest) {
        long userid = microscopicPortraitsRequest.getUserid();
        Optional<ArrayList<Map<String, Object>>> resList = microscopicPortraitsService.getMicroscopicPortraits(userid);
        if (resList.isPresent()) {
            return ApiResponse.ok(resList); // 如果成功获取到微观画像，则返回成功的响应
        } else {
            return ApiResponse.error("Product not found for Goods Description: " + userid); // 根据userid搜索画像但没搜到，返回报错信息
        }
    }
}
