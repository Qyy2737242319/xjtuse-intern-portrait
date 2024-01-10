package com.example.tags_web_back.model;

import lombok.Data;
import java.io.Serializable;

@Data
public class User_tags implements Serializable{
    private Long userid;
    private Long tagsid;
}
