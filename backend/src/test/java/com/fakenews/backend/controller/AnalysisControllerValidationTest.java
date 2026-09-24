package com.fakenews.backend.controller;

import static org.hamcrest.Matchers.containsString;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import com.fakenews.backend.service.AnalysisService;

/**
 * Request-validation tests for POST /api/v1/analyze and GET /api/v1/history.
 *
 * {@code @WebMvcTest} loads only the web layer (controller + the
 * {@code @RestControllerAdvice} exception handler), so {@link AnalysisService}
 * is a Mockito mock that is never actually invoked here -- these tests check
 * Bean Validation and error-JSON shape, not business logic.
 */
@WebMvcTest(AnalysisController.class)
class AnalysisControllerValidationTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private AnalysisService analysisService;

    @Test
    void emptyTextIsRejectedWithBadRequest() throws Exception {
        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("validation_failed"))
                .andExpect(jsonPath("$.message").exists());

        verifyNoInteractions(analysisService);
    }

    @Test
    void blankTextIsRejectedWithBadRequest() throws Exception {
        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"   \"}"))
                .andExpect(status().isBadRequest());

        verifyNoInteractions(analysisService);
    }

    @Test
    void oversizedTextIsRejectedWithBadRequest() throws Exception {
        String tooLong = "a".repeat(20_001);

        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"" + tooLong + "\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.message", containsString("20000")));

        verifyNoInteractions(analysisService);
    }

    @Test
    void unknownModelIsRejectedWithBadRequest() throws Exception {
        mockMvc.perform(post("/api/v1/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"text\": \"a perfectly normal article\", \"model\": \"gpt5\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("invalid_model"));

        verifyNoInteractions(analysisService);
    }

    @Test
    void missingBodyIsRejectedWithBadRequestNotStackTrace() throws Exception {
        mockMvc.perform(post("/api/v1/analyze").contentType(MediaType.APPLICATION_JSON))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("malformed_request"));
    }

    @Test
    void historyLimitOutOfRangeIsRejectedWithBadRequest() throws Exception {
        mockMvc.perform(get("/api/v1/history").param("limit", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("validation_failed"));

        mockMvc.perform(get("/api/v1/history").param("limit", "101"))
                .andExpect(status().isBadRequest());
    }
}
