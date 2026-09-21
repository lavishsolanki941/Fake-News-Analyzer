package com.fakenews.backend.dto;

/** One word/phrase and how hard it pushed the model (positive = toward the predicted label). */
public record WordContribution(String word, double contribution) {
}
