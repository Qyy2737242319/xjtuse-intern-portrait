package com.example.tags_web_back.service;

import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;

public interface MicroscopicPortraitsService {
    Optional<ArrayList<Map<String, Object>>> getMicroscopicPortraits(long userid);
}
