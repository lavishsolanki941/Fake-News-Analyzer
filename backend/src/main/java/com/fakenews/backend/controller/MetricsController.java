package com.fakenews.backend.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.fakenews.backend.service.MlServiceClient;
import com.fasterxml.jackson.databind.JsonNode;

@RestController
@RequestMapping("/api/v1")
public class MetricsController {

    private final MlServiceClient mlClient;

    public MetricsController(MlServiceClient mlClient) {
        this.mlClient = mlClient;
    }

    /** Proxies the ML service's metrics.json as-is (its field names are already snake_case). */
    @GetMapping("/metrics")
    public JsonNode metrics() {
        return mlClient.metrics();
    }
}
