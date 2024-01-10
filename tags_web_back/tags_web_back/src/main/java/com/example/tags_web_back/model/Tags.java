package com.example.tags_web_back.model;

import lombok.Data;
import java.io.Serializable;

@Data
public class Tags implements Serializable {
    private int tag_id;
    private int tag_level;
    private String tag_des;
    private int tag_pid;
}
