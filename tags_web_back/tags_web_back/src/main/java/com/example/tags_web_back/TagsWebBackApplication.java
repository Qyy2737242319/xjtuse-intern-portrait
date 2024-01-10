package com.example.tags_web_back;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(basePackages = {"com.example.tags_web_back.dao"})
public class TagsWebBackApplication {

    public static void main(String[] args) {
        SpringApplication.run(TagsWebBackApplication.class, args);
    }

}
