package com.fakenews.backend.dto;

/** ML service /predict response. {@code probability} is model confidence, not truth. */
public record MlPredictResponse(String label, double probability, String model) {
}
