# Mythri SLM — Project Guide

This guide explains the project in simple English. Each directory has its own `README.md`. The guides use five questions: **What? Why? Where? When? Who or what uses it?**

## What is this project?

Mythri SLM is a small model layer intended to read a message and estimate signals such as emotion and risk. The inspected source contains a text encoder, prediction heads, training tools, evaluation tools, and a separate rule-based safety layer. The command-line program in `main.py` connects prediction with safety checks and an interactive demo.

This is a description of the supplied source snapshot, not a claim that every planned product feature is implemented or clinically validated.

## How a message is processed

1. `data_utils/tokenizer.py` converts text into token IDs and attention masks.
2. `inference/predictor.py` prepares a request and calls the model.
3. `model/cognitive_slm.py` combines the text encoder and prediction heads.
4. The emotion head applies sigmoid to each emotion score. The risk head applies softmax across the three risk classes.
5. `safety/deterministic_policy.py` checks message text against explicit safety patterns and applies rule-based decisions. This is a separate layer from the learned model.
6. `main.py` presents results through its diagnostic display or interactive loop.

The exact runtime path depends on the selected entry point and configuration. Read each directory guide for its role and limitations.

## Directory guide

| Directory / file group | Guide | Main purpose |
|---|---|---|
| `model/` | [Model guide](model/README.md) | Encoder, prediction heads, conversation state |
| `inference/` | [Inference guide](inference/README.md) | Load model and predict from text |
| `safety/` | [Safety guide](safety/README.md) | Explicit text-pattern safety checks |
| `training/` | [Training guide](training/README.md) | Losses, training loop, early stopping |
| `dataset/` | [Dataset guide](dataset/README.md) | Prepare examples and build training batches |
| `data_utils/` | [Data utilities guide](data_utils/README.md) | Tokenization and synthetic examples |
| `evaluation/` | [Evaluation guide](evaluation/README.md) | Metrics, calibration, plots |
| `experiments/` | [Experiments guide](experiments/README.md) | Experiment setup and recorded results |
| `configs/` | [Configuration guide](configs/README.md) | Model and training settings |
| `tests/` | [Tests guide](tests/README.md) | Checks for selected cognitive issues |
| `codeql/` | [CodeQL guide](codeql/README.md) | Source-code audit scripts and reports |
| `codeql-custom-queries-actions/` | [Custom queries guide](codeql-custom-queries-actions/README.md) | CodeQL query-pack files |
| `model_trining/` | [Training entry-point guide](model_trining/README.md) | A run script found in the supplied snapshot |
| `founder/` | [Founder materials guide](founder/README.md) | Panel and planning material; handle access carefully |

## Mathematics in plain English

### Emotion scores

For each emotion score `z`, the source applies the sigmoid function:

\[
p = \frac{1}{1 + e^{-z}}
\]

This converts any real-valued score into a number between 0 and 1. For example, when `z = 0`, `p = 0.5`. The implementation treats emotion labels independently, so several emotions can have high scores at once. A score is not automatically a well-calibrated real-world probability.

### Risk classes

The risk head applies softmax to the class scores:

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

This makes the three class scores add up to 1. The predicted class is the index with the largest raw score. The code also returns the probability assigned to the final class, which is treated as the risk class in this implementation.

### Class-balanced loss

Training loss gives different importance to examples from different classes. In the focal-loss implementation, the factor `(1 - p_t)^gamma` reduces the contribution from examples the model already handles confidently. `alpha` and class weights can increase the importance of selected classes. The exact loss selected depends on the experiment configuration and the trainer's implementation.

## Important source-snapshot notes

- The supplied bundle contains `model_trining/run_experiment.py`, while the Hugging Face file listing previously provided showed `run_experiment.py` at the repository root. Confirm which version is current before renaming or moving either file.
- The YAML comments and values should not be assumed to describe every active model detail. The pretrained DistilBERT encoder has its own architecture; inspect the active constructor and runtime path before treating YAML architecture fields as authoritative.
- Some heads and features may exist in code but not be trained or used by the baseline experiment. Documentation must distinguish “defined”, “trained”, and “used at inference”.
- The source bundle is a snapshot. No application files have been changed by preparing these guides.

## Safe change policy

Before renaming a file, search for imports, path strings, CLI commands, tests, and checkpoint/config references. Rename only after making a dependency map. Then run tests and a small inference smoke test. This documentation pack does not itself rename project files.
# SLM
# mythri_slm
