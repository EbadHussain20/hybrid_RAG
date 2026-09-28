# Evaluation Report

Dataset: `eval_questions.json` (30 questions)

## Retrieval: Dense vs Sparse vs Hybrid

| Metric | Dense Only | Sparse (BM25) Only | Hybrid (RRF) |
|---|---|---|---|
| recall@1 | 0.75 | 0.731 | 0.712 |
| recall@3 | 0.923 | 0.923 | 0.942 |
| recall@5 | 0.962 | 1.0 | 0.981 |
| precision@1 | 0.846 | 0.846 | 0.808 |
| precision@3 | 0.628 | 0.59 | 0.603 |
| precision@5 | 0.485 | 0.438 | 0.5 |
| mrr | 0.904 | 0.904 | 0.886 |
| hit_rate | 0.962 | 1.0 | 1.0 |

(Evaluated over 26 questions that have ground-truth expected sources; no-answer questions are excluded from retrieval metrics since they have no ground truth to retrieve.)

## Generation: Citation & Grounding Quality

- Citation validity rate: **100%** (fraction of answers whose citation markers all map to real retrieved chunks)
- Groundedness rate: **100%** (answers whose vocabulary is substantially supported by retrieved context)
- No-answer accuracy: **50%** (correctly declined to answer when no evidence exists)
- Questions evaluated: 30

## Per-Question Results

| ID | Category | Citations Valid | Grounded | No-Answer (expected/produced) | Latency (ms) |
|---|---|---|---|---|---|
| q01 | direct_factual | True | True | False/False | 9.5 |
| q02 | direct_factual | True | True | False/False | 8.25 |
| q03 | direct_factual | True | True | False/False | 5.44 |
| q04 | direct_factual | True | True | False/False | 8.86 |
| q05 | direct_factual | True | True | False/False | 7.76 |
| q06 | semantic | True | True | False/True | 7.06 |
| q07 | semantic | True | True | False/True | 10.54 |
| q08 | semantic | True | True | False/False | 5.41 |
| q09 | semantic | True | True | False/False | 5.44 |
| q10 | semantic | True | True | False/False | 5.85 |
| q11 | keyword_heavy | True | True | False/False | 6.04 |
| q12 | keyword_heavy | True | True | False/False | 5.41 |
| q13 | keyword_heavy | True | True | False/False | 5.47 |
| q14 | keyword_heavy | True | True | False/False | 6.37 |
| q15 | keyword_heavy | True | True | False/False | 7.0 |
| q16 | multi_document | True | True | False/False | 7.01 |
| q17 | multi_document | True | True | False/False | 7.71 |
| q18 | multi_document | True | True | False/False | 5.24 |
| q19 | multi_document | True | True | False/False | 6.75 |
| q20 | multi_document | True | True | False/False | 6.35 |
| q21 | synthesis | True | True | False/False | 7.01 |
| q22 | synthesis | True | True | False/False | 7.68 |
| q23 | synthesis | True | True | False/True | 6.66 |
| q24 | distractor | True | True | False/True | 6.62 |
| q25 | distractor | True | True | False/False | 6.1 |
| q26 | no_answer | True | True | True/True | 6.57 |
| q27 | no_answer | True | True | True/False | 8.33 |
| q28 | no_answer | True | True | True/False | 6.52 |
| q29 | edge_case | True | True | False/False | 5.64 |
| q30 | edge_case | True | True | True/True | 4.01 |