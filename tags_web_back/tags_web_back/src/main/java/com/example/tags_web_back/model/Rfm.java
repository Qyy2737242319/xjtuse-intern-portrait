package com.example.tags_web_back.model;

import lombok.Data;
import java.io.Serializable;

@Data
public class Rfm implements Serializable{
    private Double R_score;
    private Double F_score;
    private Double M_score;
    private String R;
    private String F;
    private String M;
    private String level;
    private int id;
}
