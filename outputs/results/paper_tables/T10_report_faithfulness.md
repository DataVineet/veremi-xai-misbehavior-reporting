Table T10. Automatic faithfulness validation of generated reports on a class-stratified test sample.

| requested_backend | model | n_reports | written_by_llm | llm_passed_first_try | llm_passed_after_retry | llm_rejected_twice_fallback_to_template | validation_pass_rate | mean_words | median_analyse_s (features+model+SHAP) | median_report_s | misclassified_in_sample | low_or_moderate_confidence_in_sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| template | deterministic-template | 200 | 0 | 0 | 0 | 0 | 1.0000 | 328.0000 | 0.1070 | 0.0060 | 96 | 79 |
| groq_small | openai/gpt-oss-20b | 60 | 60 | 55 | 5 | 0 | 1.0000 | 256.9000 | 0.0740 | 1.6360 | 20 | 20 |
| groq | openai/gpt-oss-120b | 60 | 59 | 55 | 4 | 1 | 1.0000 | 304.2000 | 0.2240 | 4.3150 | 20 | 20 |
