package com.fakenews.backend.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.fakenews.backend.entity.PredictionRecord;

public interface PredictionRecordRepository extends JpaRepository<PredictionRecord, Long> {
}
