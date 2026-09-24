package com.fakenews.backend.controller;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import com.fakenews.backend.dto.MlExplainResponse;
import com.fakenews.backend.dto.MlPredictResponse;
import com.fakenews.backend.exception.ApiException;
import com.fakenews.backend.repository.PredictionRecordRepository;
import com.fakenews.backend.service.MlServiceClient;

/**
 * Full Spring context (real AnalysisService, real H2 repository via the
 * "test" profile's in-memory database) with only {@link MlServiceClient}
 * mocked. This lets these tests exercise the happy path, error mapping, and
 * history persistence for real, without ever needing an actual Python ML
 * service running -- MlServiceClient is the ONLY thing standing in for it.
 */
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class AnalysisServiceIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private PredictionRecordRepository repository;

    @MockitoBean
    private MlServiceClient mlServiceClient;

    @Test
    void analyzeHappyPathReturnsCombinedResponse() throws Exception {
        given(mlServiceClient.predict(anyString(), anyString()))
                .willReturn(new MlPredictResponse("REAL", 0.87, "logistic_regression"));
        given(mlServiceClient.explain(anyString(), anyString()))
                .willReturn(new MlExplainResponse("logistic_regression", List.of(), List.of(), "note"));

        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"a perfectly normal news article body\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.label").value("REAL"))
                .andExpect(jsonPath("$.probability").value(0.87))
                .andExpect(jsonPath("$.model").value("logistic_regression"))
                .andExpect(jsonPath("$.explanation.note").value("note"))
                .andExpect(jsonPath("$.disclaimer").exists());
    }

    @Test
    void mlServiceUnreachableMapsToCleanServiceUnavailableJson() throws Exception {
        // Simulates what MlServiceClient itself throws when the ML service is
        // down (see MlServiceClient.execute() catching ResourceAccessException).
        given(mlServiceClient.predict(anyString(), anyString()))
                .willThrow(new ApiException(HttpStatus.SERVICE_UNAVAILABLE, "ml_service_unavailable",
                        "The ML service is currently unreachable or too slow to respond. Please try again shortly."));

        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"a perfectly normal news article body\"}"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").value("ml_service_unavailable"))
                .andExpect(jsonPath("$.message").exists());
    }

    @Test
    void unexpectedExceptionFromMlClientMapsToInternalErrorNotStackTrace() throws Exception {
        given(mlServiceClient.predict(anyString(), anyString()))
                .willThrow(new RuntimeException("boom"));

        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"a perfectly normal news article body\"}"))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.error").value("internal_error"))
                .andExpect(jsonPath("$.message").value("Something went wrong on our side. Please try again."));
    }

    @Test
    void successfulAnalysisIsSavedAndReturnedFromHistory() throws Exception {
        given(mlServiceClient.predict(anyString(), anyString()))
                .willReturn(new MlPredictResponse("FAKE", 0.62, "multinomial_nb"));
        given(mlServiceClient.explain(anyString(), anyString()))
                .willReturn(new MlExplainResponse("multinomial_nb", List.of(), List.of(), "note"));

        long before = repository.count();

        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"history persistence test article\", \"model\": \"multinomial_nb\"}"))
                .andExpect(status().isOk());

        assertThat(repository.count()).isEqualTo(before + 1);

        mockMvc.perform(get("/api/v1/history").param("limit", "5"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].label").value("FAKE"))
                .andExpect(jsonPath("$[0].model").value("multinomial_nb"))
                .andExpect(jsonPath("$[0].text_snippet").value("history persistence test article"));
    }
}
