package com.fakenews.backend.dto;

import java.util.List;

/** The "words that pushed the model" part of an analyze response. */
public record Explanation(
        List<WordContribution> topPositive,
        List<WordContribution> topNegative,
        String note) {
}
