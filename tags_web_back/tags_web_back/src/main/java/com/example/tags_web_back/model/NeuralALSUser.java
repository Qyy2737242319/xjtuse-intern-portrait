
package com.example.tags_web_back.model;

import lombok.Data;

import java.io.Serializable;

// 定义一个用户实体类，用于封装数据库中的用户和商品的数据
@Data
public class NeuralALSUser implements Serializable{

    // 定义用户和商品的属性

    private int pid; // 用户id
    private int id; // 商品id
    private double pre; // 预测用户满意度
}