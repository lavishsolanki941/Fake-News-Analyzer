package com.fakenews.backend.dto;

import java.time.Instant;

import com.fakenews.backend.entity.PredictionRecord;

/** One row of GET /api/v1/history. */
public record HistoryItem(
        Long id,
        String textSnippet,
        String model,
        String label,
        double probability,
        Instant createdAt) {

    public static HistoryItem from(PredictionRecord record) {
        return new HistoryItem(
                record.getId(),
                record.getTextSnippet(),
                record.getModel(),
                record.getLabel(),
                record.getProbability(),
                record.getCreatedAt());
    }
}
