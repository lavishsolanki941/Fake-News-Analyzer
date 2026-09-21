package com.fakenews.backend.controller;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.fakenews.backend.dto.AnalyzeRequest;
import com.fakenews.backend.dto.AnalyzeResponse;
import com.fakenews.backend.dto.HistoryItem;
import com.fakenews.backend.exception.ApiException;
import com.fakenews.backend.service.AnalysisService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/v1")
public class AnalysisController {

    private static final int DEFAULT_HISTORY_LIMIT = 20;
    private static final int MAX_HISTORY_LIMIT = 100;

    private final AnalysisService analysisService;

    public AnalysisController(AnalysisService analysisService) {
        this.analysisService = analysisService;
    }

    @PostMapping("/analyze")
    public AnalyzeResponse analyze(@Valid @RequestBody AnalyzeRequest request) {
        return analysisService.analyze(request);
    }

    @GetMapping("/history")
    public List<HistoryItem> history(@RequestParam(defaultValue = "" + DEFAULT_HISTORY_LIMIT) int limit) {
        if (limit < 1 || limit > MAX_HISTORY_LIMIT) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "validation_failed",
                    "limit must be between 1 and " + MAX_HISTORY_LIMIT + ".");
        }
        return analysisService.recent(limit);
    }
}
