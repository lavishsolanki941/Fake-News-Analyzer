package com.fakenews.backend.dto;

import java.util.List;

/** ML service /health response. */
public record MlHealthResponse(String status, boolean trained, List<String> modelsAvailable) {
}
