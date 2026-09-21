package com.fakenews.backend.dto;

/** Body sent to the ML service's /predict and /explain. */
public record MlRequest(String text, String model) {
}
