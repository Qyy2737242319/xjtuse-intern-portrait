package com.example.tags_web_back.utils;

import com.example.tags_web_back.model.Tags;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ListToTree {

    public static List<Map<String, Object>> listToTree(List<Tags> dataList, int root, String rootField, String nodeField) {
        List<Map<String, Object>> respList = new ArrayList<>();

        for (Tags tag : dataList) {
            if (tag.getTag_pid() == root) {
                Map<String, Object> node = new HashMap<>();
                node.put("tag_id", tag.getTag_id());
                node.put("tag_level", tag.getTag_level());
                node.put("tag_des", tag.getTag_des());
                node.put("tag_pid", tag.getTag_pid());

                List<Map<String, Object>> children = listToTree(dataList, tag.getTag_id(), rootField, nodeField);
                if (!children.isEmpty()) {
                    node.put("children", children);
                }
                respList.add(node);
            }
        }
        return respList;
    }
}

