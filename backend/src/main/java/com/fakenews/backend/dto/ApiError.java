package com.fakenews.backend.dto;

/** Every error response body: a short machine-readable code and a human-readable message. */
public record ApiError(String error, String message) {
}
