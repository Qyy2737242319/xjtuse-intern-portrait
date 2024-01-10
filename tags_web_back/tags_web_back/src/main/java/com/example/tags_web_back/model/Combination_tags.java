package com.example.tags_web_back.model;

import lombok.Data;
import java.io.Serializable;

@Data
public class Combination_tags implements Serializable{
    private int tag_id;
    private String tag_name;
    private String tag_des;
    private int tag_level;
}
