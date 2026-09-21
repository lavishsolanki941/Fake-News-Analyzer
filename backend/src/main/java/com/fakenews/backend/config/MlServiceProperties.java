package com.fakenews.backend.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * Settings for reaching the Python ML service (property prefix {@code ml.service}).
 * The URL comes from the ML_SERVICE_URL env var, see application.yml.
 */
@ConfigurationProperties(prefix = "ml.service")
public record MlServiceProperties(
        @DefaultValue("http://localhost:8000") String url,
        @DefaultValue("2s") Duration connectTimeout,
        @DefaultValue("10s") Duration readTimeout) {
}
