package com.fakenews.backend.service;

import java.io.IOException;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import com.fakenews.backend.dto.MlExplainResponse;
import com.fakenews.backend.dto.MlHealthResponse;
import com.fakenews.backend.dto.MlPredictResponse;
import com.fakenews.backend.dto.MlRequest;
import com.fakenews.backend.exception.ApiException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * The only class that talks to the Python ML service. It maps the service's
 * JSON to Java objects and turns every failure (down, slow, rejected input,
 * unexpected reply) into an {@link ApiException} with a client-safe message.
 */
@Service
public class MlServiceClient {

    private static final Logger log = LoggerFactory.getLogger(MlServiceClient.class);

    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    public MlServiceClient(RestClient mlRestClient, ObjectMapper objectMapper) {
        this.restClient = mlRestClient;
        this.objectMapper = objectMapper;
    }

    public MlPredictResponse predict(String text, String model) {
        return post("/predict", new MlRequest(text, model), MlPredictResponse.class);
    }

    public MlExplainResponse explain(String text, String model) {
        return post("/explain", new MlRequest(text, model), MlExplainResponse.class);
    }

    /** The ML service's metrics.json, passed through untouched. */
    public JsonNode metrics() {
        return get("/metrics", JsonNode.class);
    }

    public MlHealthResponse health() {
        return get("/health", MlHealthResponse.class);
    }

    private <T> T post(String path, Object body, Class<T> type) {
        return execute(path, () -> restClient.post()
                .uri(path)
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .onStatus(HttpStatusCode::isError, this::throwForErrorStatus)
                .body(type));
    }

    private <T> T get(String path, Class<T> type) {
        return execute(path, () -> restClient.get()
                .uri(path)
                .retrieve()
                .onStatus(HttpStatusCode::isError, this::throwForErrorStatus)
                .body(type));
    }

    private <T> T execute(String path, Supplier<T> call) {
        try {
            T result = call.get();
            if (result == null) {
                throw badGateway(path, "empty response");
            }
            return result;
        } catch (ResourceAccessException ex) {
            // Connection refused, DNS failure, connect timeout, read timeout...
            log.warn("ML service unreachable or too slow for {}: {}", path, ex.getMessage());
            throw new ApiException(HttpStatus.SERVICE_UNAVAILABLE, "ml_service_unavailable",
                    "The ML service is currently unreachable or too slow to respond. Please try again shortly.");
        } catch (RestClientException ex) {
            // Reply arrived but could not be read as the expected JSON.
            throw badGateway(path, ex.getMessage());
        }
    }

    /** Runs for any 4xx/5xx from the ML service; always throws. */
    private void throwForErrorStatus(HttpRequest request, ClientHttpResponse response) throws IOException {
        int status = response.getStatusCode().value();
        String detail = extractDetail(response);
        String path = request.getURI().getPath();

        if (status == 400 || status == 422) {
            // The ML service judged the text unusable (too short, no known words, ...).
            throw new ApiException(HttpStatus.BAD_REQUEST, "invalid_input",
                    detail != null ? detail : "The ML service could not analyze this text.");
        }
        if (status == 503) {
            // The ML service is up but has no trained model / could not load it.
            log.warn("ML service not ready for {}: {}", path, detail);
            throw new ApiException(HttpStatus.SERVICE_UNAVAILABLE, "ml_service_not_ready",
                    "The ML service is running but not ready" + (detail != null ? ": " + detail : "."));
        }
        throw badGateway(path, "HTTP " + status + (detail != null ? " - " + detail : ""));
    }

    /** The ML service's errors look like {"detail": "..."}. Returns null if the body isn't that. */
    private String extractDetail(ClientHttpResponse response) {
        try {
            JsonNode detail = objectMapper.readTree(response.getBody()).get("detail");
            return detail != null && detail.isTextual() ? detail.asText() : null;
        } catch (IOException | RuntimeException ex) {
            return null;
        }
    }

    private ApiException badGateway(String path, String reason) {
        log.error("Unexpected ML service reply for {}: {}", path, reason);
        return new ApiException(HttpStatus.BAD_GATEWAY, "ml_service_error",
                "The ML service returned an unexpected response.");
    }
}
