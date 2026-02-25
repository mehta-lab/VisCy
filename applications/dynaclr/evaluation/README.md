# DynaCLR Evaluation

Evaluation tools for DynaCLR cell embedding models. Each evaluation method lives in its own subdirectory.

## Available Methods

| Method | Directory | Description |
|--------|-----------|-------------|
| Linear classifiers | `linear_classifiers/` | Logistic regression on embeddings for supervised cell phenotyping |
| Temporal smoothness | `benchmarking/smoothness/` | Evaluate how smoothly embeddings change across adjacent time frames |
