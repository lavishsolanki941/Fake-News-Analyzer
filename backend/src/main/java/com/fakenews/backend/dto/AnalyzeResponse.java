package com.fakenews.backend.dto;

/** Combined prediction + explanation returned by POST /api/v1/analyze. */
public record AnalyzeResponse(
        String label,
        double probability,
        String model,
        Explanation explanation,
        String disclaimer) {
}
