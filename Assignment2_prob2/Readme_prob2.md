# Character-Level Indian Name Generation

## Overview

This project implements and compares three character-level sequence models for generating Indian-style full names:

* Vanilla RNN
* Bidirectional LSTM (BiLSTM)
* RNN with Attention

The goal is to understand how different architectures perform in learning phonetic patterns and generating realistic names using a fixed Unicode character vocabulary.

---

## Dataset

* Source: `training_names.txt`
* Contains ~1000 Indian full names (first name + space + last name)
* Each sample is appended with a special end-of-sequence (EOS) token `.`
* Vocabulary includes all unique characters (including space) → total size: **54**

---

## Preprocessing

* Sequences are padded to the maximum length
* Padding index is masked during loss computation
* Targets are created using a one-step shift (autoregressive setup)

---

## Models

### 1. Vanilla RNN

* Architecture: Embedding → RNN → Linear
* Hidden size: 128
* Parameters: 46,902
* Learns basic phonetic transitions but struggles with long coherence

### 2. BiLSTM

* Architecture: Embedding → Bidirectional LSTM → Linear
* Hidden size: 128 (per direction) → 256 combined
* Parameters: ~284,982
* Lower training loss but weaker generation due to training–inference mismatch

### 3. Attention RNN

* Architecture: Embedding → RNN → Attention → Linear
* Attention: Linear(128 → 1) + Softmax
* Parameters: ~47,031
* Produces more coherent and realistic names

---

## Training

* Optimizer: Adam
* Learning rate: 0.003
* Epochs:

  * RNN & Attention: 40
  * BiLSTM: 20
* Loss: Cross-entropy (with padding mask)

---

## Generation Process

* Start from a random uppercase character
* Use temperature-based multinomial sampling
* Insert space to enforce first name + last name structure
* Stop generation at EOS token `.`

### Filtering Rules

* Must contain at least 2 words
* Each word length > 2
* Total length > 6

---

## Evaluation Metrics

### Novelty

* % of generated names not present in training data

### Diversity

* % of unique names among accepted outputs

* Evaluation done on 500 generated samples using `generate_eval_pool`

---

## Results Summary

| Model       | Accepted Samples | Observations                            |
| ----------- | ---------------- | --------------------------------------- |
| Vanilla RNN | 397              | Basic patterns, sometimes noisy endings |
| BiLSTM      | 166              | High diversity but less coherent        |
| Attention   | 330              | Best balance of quality and realism     |

---

## Key Insights

* Attention improves coherence by focusing on relevant past characters
* BiLSTM suffers during generation due to bidirectional training mismatch
* Vanilla RNN is simple but limited in long-sequence consistency
* Lower training loss ≠ better generation quality
* Quantitative metrics alone are insufficient → qualitative analysis is essential

---

## Common Issues

* Inconsistent capitalization (partially fixed with `clean_sample`)
* Long or unfinished sequences when EOS probability is low
* Repeated characters and elongated surnames
* Attention instability (reduced via time-wise pooling)

---

## Conclusion

The Attention RNN performs best overall, generating names that are both diverse and phonetically consistent. This project highlights the importance of aligning training and inference strategies and combining both qualitative and quantitative evaluation for generative models.

---

## How to Run

1. Load dataset from `training_names.txt`
2. Preprocess sequences and build vocabulary
3. Train models (RNN, BiLSTM, Attention)
4. Generate samples using temperature sampling
5. Evaluate using novelty and diversity metrics

---

## Future Improvements

* Better EOS calibration for cleaner stopping
* Beam search instead of random sampling
* Larger dataset for improved generalization
* Transformer-based architecture for comparison

---
