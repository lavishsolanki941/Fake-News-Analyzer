package com.fakenews.backend.service;

import org.springframework.stereotype.Service;

import com.fakenews.backend.dto.HealthResponse;
import com.fakenews.backend.dto.MlHealthResponse;
import com.fakenews.backend.dto.MlServiceHealth;
import com.fakenews.backend.exception.ApiException;

@Service
public class HealthService {

    private final MlServiceClient mlClient;

    public HealthService(MlServiceClient mlClient) {
        this.mlClient = mlClient;
    }

    /**
     * Never throws: if the ML service is down that IS the answer, reported as
     * DEGRADED rather than as an error, so a status page can still render.
     */
    public HealthResponse check() {
        try {
            MlHealthResponse ml = mlClient.health();
            String message = ml.trained() ? null : "The ML service is running but has no trained model. Run `python -m src.train` in ml-service.";
            MlServiceHealth mlHealth = new MlServiceHealth("UP", ml.trained(), ml.modelsAvailable(), message);
            return new HealthResponse(ml.trained() ? "UP" : "DEGRADED", "UP", mlHealth);
        } catch (ApiException ex) {
            return new HealthResponse("DEGRADED", "UP", new MlServiceHealth("DOWN", null, null, ex.getMessage()));
        }
    }
}
