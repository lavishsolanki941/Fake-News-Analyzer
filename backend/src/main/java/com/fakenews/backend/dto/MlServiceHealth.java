package com.fakenews.backend.dto;

import java.util.List;

/** What the backend knows about the ML service. {@code trained}/{@code modelsAvailable} are null when it's DOWN. */
public record MlServiceHealth(String status, Boolean trained, List<String> modelsAvailable, String message) {
}
