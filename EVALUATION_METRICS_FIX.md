# Evaluation Metrics Fix

## Problem
The Evaluate page was displaying:
- All accuracy metrics as 0.0%
- NaN% values for retrieval metrics

## Root Cause
1. **Backend Missing Metrics**: The `/api/rag/evaluate` endpoint was only returning LLM-as-judge metrics, not the accuracy and retrieval metrics
2. **Frontend Defensive Missing**: The frontend tried to access `res.exact_match`, `res.f1`, `res.recall_10`, etc., but these weren't in the response
3. **NaN Handling**: No defensive checks for NaN values in formatters

## Changes Made

### Backend (`app/rag/evaluate.py`)
- Updated `evaluate_rag()` to return all metrics at the top level:
  - Accuracy methods: `exact_match`, `semantic_similarity`, `f1`, `accuracy_combined`
  - Retrieval metrics: `recall_10`, `recall_20`, `recall_50`, `mrr`, `ndcg_10`
  - Backward compatibility: included `accuracy` alias for `accuracy_llm`

### Backend (`app/evaluation/agent_runner.py`)
- Updated `evaluate_question()` to return all individual metrics from the evaluator
- Updated `failed_question_row()` to initialize all metric fields with 0.0

### Frontend (`pages/Evaluate.tsx`)
- Updated `GradeChip` to handle NaN scores defensively

### Frontend (`components/ScoreBar.tsx`)
- Added NaN check before calculation: `isNaN(score) ? 0 : score`

### Frontend (`utils/format.ts`)
- Updated `formatScore()` to handle NaN gracefully

## Result
- Accuracy metrics now display correctly with real percentages
- Retrieval metrics now display correctly with real percentages  
- No more NaN values
- Backend returns complete evaluation data
- Frontend handles edge cases safely
