package com.fakenews.backend.exception;

import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.TypeMismatchException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import com.fakenews.backend.dto.ApiError;

/**
 * Turns every failure into {@code {"error": "...", "message": "..."}}.
 * Extending ResponseEntityExceptionHandler means Spring's own MVC errors
 * (404, 405, bad JSON, ...) also pass through handleExceptionInternal below
 * instead of leaking Spring's default error bodies.
 */
@RestControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /** Errors we raised on purpose (ML service down, ML rejected input, bad query param, ...). */
    @ExceptionHandler(ApiException.class)
    public ResponseEntity<ApiError> handleApiException(ApiException ex) {
        return ResponseEntity.status(ex.getStatus()).body(new ApiError(ex.getError(), ex.getMessage()));
    }

    /** Anything unexpected: log it server-side, tell the client nothing internal. */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> handleUnexpected(Exception ex) {
        log.error("Unhandled exception", ex);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiError("internal_error", "Something went wrong on our side. Please try again."));
    }

    @Override
    protected ResponseEntity<Object> handleMethodArgumentNotValid(
            MethodArgumentNotValidException ex, HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        boolean modelInvalid = ex.getBindingResult().getFieldErrors().stream()
                .anyMatch(e -> "model".equals(e.getField()));
        String message = ex.getBindingResult().getFieldErrors().stream()
                .map(FieldError::getDefaultMessage)
                .sorted()
                .collect(Collectors.joining("; "));
        return body(HttpStatus.BAD_REQUEST, modelInvalid ? "invalid_model" : "validation_failed", message);
    }

    @Override
    protected ResponseEntity<Object> handleHttpMessageNotReadable(
            HttpMessageNotReadableException ex, HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        return body(HttpStatus.BAD_REQUEST, "malformed_request",
                "Request body is missing or is not valid JSON. Expected: {\"text\": \"...\", \"model\": \"...\"}.");
    }

    @Override
    protected ResponseEntity<Object> handleTypeMismatch(
            TypeMismatchException ex, HttpHeaders headers, HttpStatusCode status, WebRequest request) {
        return body(HttpStatus.BAD_REQUEST, "validation_failed",
                "Parameter '" + ex.getPropertyName() + "' has an invalid value.");
    }

    /** Fallback for all other Spring MVC errors (404, 405, 415, missing params, ...). */
    @Override
    protected ResponseEntity<Object> handleExceptionInternal(
            Exception ex, Object body, HttpHeaders headers, HttpStatusCode statusCode, WebRequest request) {
        HttpStatus status = HttpStatus.valueOf(statusCode.value());
        String message = switch (status) {
            case NOT_FOUND -> "No such endpoint.";
            case METHOD_NOT_ALLOWED -> "This HTTP method is not supported for this endpoint.";
            case UNSUPPORTED_MEDIA_TYPE -> "Unsupported content type. Send JSON with Content-Type: application/json.";
            case NOT_ACCEPTABLE -> "This endpoint only produces JSON.";
            default -> "The request could not be processed.";
        };
        return body(status, status.getReasonPhrase().toLowerCase().replace(' ', '_'), message);
    }

    private static ResponseEntity<Object> body(HttpStatus status, String error, String message) {
        return ResponseEntity.status(status).body(new ApiError(error, message));
    }
}
