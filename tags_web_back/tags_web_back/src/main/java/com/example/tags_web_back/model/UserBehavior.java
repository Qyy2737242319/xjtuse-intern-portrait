package com.example.tags_web_back.model;

import lombok.Data;
import java.io.Serializable;

@Data
public class UserBehavior implements Serializable{
    private String user_id;
    private String item_id;
    private String category_id;
    private String behavior_type;
    private String timestamp;
}
