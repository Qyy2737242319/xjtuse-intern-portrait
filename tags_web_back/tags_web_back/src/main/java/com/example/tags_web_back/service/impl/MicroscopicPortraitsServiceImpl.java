package com.example.tags_web_back.service.impl;


import com.example.tags_web_back.dao.MicroscopicPortraitsMapper;
import com.example.tags_web_back.dao.TagsMapper;
import com.example.tags_web_back.model.Tags;
import com.example.tags_web_back.service.MicroscopicPortraitsService;
import com.example.tags_web_back.utils.ListToTree;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;

@Service
public class MicroscopicPortraitsServiceImpl implements MicroscopicPortraitsService {
    final MicroscopicPortraitsMapper microscopicPortraitsMapper;
    final TagsMapper tagsMapper;

    public MicroscopicPortraitsServiceImpl(@Autowired MicroscopicPortraitsMapper microscopicPortraitsMapper,
                                           @Autowired TagsMapper tagsMapper) {
        this.microscopicPortraitsMapper = microscopicPortraitsMapper;
        this.tagsMapper = tagsMapper;
    }

    @Override
    public Optional<ArrayList<Map<String, Object>>> getMicroscopicPortraits(long userid) {
        // 获取想查用户的五级标签
        ArrayList<Tags> tags = microscopicPortraitsMapper.getJoinedTagsByUserId(userid);

        // 获得所有的1-4级标签
        ArrayList<Tags> under4LevelTags = tagsMapper.getTagsUnderLevel(4);

        // 将tags和under4LevelTags中的数据合并
        tags.addAll(under4LevelTags);

        // 组织成树结构
        ArrayList<Map<String, Object>> resultList = (ArrayList<Map<String, Object>>) ListToTree.listToTree(tags, -1, "tag_pid", "tag_id");

        // 处理完毕后，将结果作为Optional返回
        return Optional.of(resultList);
    }

}
