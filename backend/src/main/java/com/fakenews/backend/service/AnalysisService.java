package com.fakenews.backend.service;

import java.util.List;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import com.fakenews.backend.dto.AnalyzeRequest;
import com.fakenews.backend.dto.AnalyzeResponse;
import com.fakenews.backend.dto.Explanation;
import com.fakenews.backend.dto.HistoryItem;
import com.fakenews.backend.dto.MlExplainResponse;
import com.fakenews.backend.dto.MlPredictResponse;
import com.fakenews.backend.entity.PredictionRecord;
import com.fakenews.backend.repository.PredictionRecordRepository;

@Service
public class AnalysisService {

    public static final String DISCLAIMER = "ML prediction, not fact-checking";

    private final MlServiceClient mlClient;
    private final PredictionRecordRepository repository;

    public AnalysisService(MlServiceClient mlClient, PredictionRecordRepository repository) {
        this.mlClient = mlClient;
        this.repository = repository;
    }

    // Deliberately NOT @Transactional: the slow ML calls must not hold a DB
    // connection, and repository.save() opens its own short transaction.
    public AnalyzeResponse analyze(AnalyzeRequest request) {
        String model = request.resolvedModel();

        MlPredictResponse prediction = mlClient.predict(request.text(), model);
        MlExplainResponse explanation = mlClient.explain(request.text(), model);

        // Only reached if both ML calls succeeded, so history never holds half-finished analyses.
        repository.save(new PredictionRecord(
                request.text(), prediction.model(), prediction.label(), prediction.probability()));

        return new AnalyzeResponse(
                prediction.label(),
                prediction.probability(),
                prediction.model(),
                new Explanation(explanation.topPositive(), explanation.topNegative(), explanation.note()),
                DISCLAIMER);
    }

    /** Most recent analyses first. */
    public List<HistoryItem> recent(int limit) {
        return repository.findAll(PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "createdAt", "id")))
                .map(HistoryItem::from)
                .getContent();
    }
}
