package com.fakenews.backend.dto;

import java.util.List;

/** ML service /explain response (snake_case JSON maps via the global naming strategy). */
public record MlExplainResponse(
        String model,
        List<WordContribution> topPositive,
        List<WordContribution> topNegative,
        String note) {
}
