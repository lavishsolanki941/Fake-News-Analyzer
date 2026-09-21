package com.fakenews.backend.exception;

import org.springframework.http.HttpStatus;

/**
 * An error we deliberately report to the API client. The message is always
 * safe to show to them; {@link GlobalExceptionHandler} turns it into JSON.
 */
public class ApiException extends RuntimeException {

    private final HttpStatus status;
    private final String error;

    public ApiException(HttpStatus status, String error, String message) {
        super(message);
        this.status = status;
        this.error = error;
    }

    public HttpStatus getStatus() {
        return status;
    }

    public String getError() {
        return error;
    }
}
