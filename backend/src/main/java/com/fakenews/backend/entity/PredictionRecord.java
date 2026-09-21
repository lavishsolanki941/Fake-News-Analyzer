package com.fakenews.backend.entity;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * One saved analysis. Only a short snippet of the text is stored, never the
 * whole article.
 */
@Entity
@Table(name = "prediction_record")
public class PredictionRecord {

    public static final int SNIPPET_LENGTH = 200;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = SNIPPET_LENGTH)
    private String textSnippet;

    @Column(nullable = false, length = 50)
    private String model;

    @Column(nullable = false, length = 10)
    private String label;

    @Column(nullable = false)
    private double probability;

    @Column(nullable = false, updatable = false)
    private Instant createdAt;

    protected PredictionRecord() {
        // required by JPA
    }

    public PredictionRecord(String text, String model, String label, double probability) {
        this.textSnippet = snippet(text);
        this.model = model;
        this.label = label;
        this.probability = probability;
        this.createdAt = Instant.now();
    }

    /** Whitespace-collapsed first {@value #SNIPPET_LENGTH} characters, without splitting an emoji in half. */
    static String snippet(String text) {
        String collapsed = text.strip().replaceAll("\\s+", " ");
        if (collapsed.length() <= SNIPPET_LENGTH) {
            return collapsed;
        }
        int end = SNIPPET_LENGTH;
        if (Character.isHighSurrogate(collapsed.charAt(end - 1))) {
            end--;
        }
        return collapsed.substring(0, end);
    }

    public Long getId() {
        return id;
    }

    public String getTextSnippet() {
        return textSnippet;
    }

    public String getModel() {
        return model;
    }

    public String getLabel() {
        return label;
    }

    public double getProbability() {
        return probability;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
