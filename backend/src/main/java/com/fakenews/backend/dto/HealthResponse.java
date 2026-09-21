package com.fakenews.backend.dto;

/**
 * GET /api/v1/health. {@code status} is UP only when the backend is up AND the ML
 * service is reachable with a trained model; otherwise DEGRADED.
 */
public record HealthResponse(String status, String backend, MlServiceHealth mlService) {
}
