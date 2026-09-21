package com.fakenews.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/** Body of POST /api/v1/analyze. {@code model} is optional. */
public record AnalyzeRequest(
        @NotBlank(message = "text must not be blank")
        @Size(max = 20000, message = "text must be at most 20000 characters")
        String text,

        // Null is allowed (means "use the default"); anything else must be a model the ML service serves.
        @Pattern(regexp = "logistic_regression|multinomial_nb",
                 message = "model must be one of: logistic_regression, multinomial_nb")
        String model) {

    public static final String DEFAULT_MODEL = "logistic_regression";

    public String resolvedModel() {
        return model == null ? DEFAULT_MODEL : model;
    }
}
