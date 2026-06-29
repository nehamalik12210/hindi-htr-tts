# Phase 3.5 — Fine-Tuning TrOCR-Small for Hindi Handwritten Text Recognition

## Overview

Phase 3.5 explores a **Transformer-based OCR (TrOCR)** approach, replacing the CRNN architecture from Phase 3 with a pretrained **ViT encoder + GPT-2 decoder** model. By fine-tuning `microsoft/trocr-small-handwritten` with a custom Devanagari tokenizer and a two-phase training strategy, this phase achieves **94.66% character accuracy** (CER = 0.0534) and **87.10% word accuracy** (WER = 0.1290) — a significant improvement over the Phase 3 CRNN results.

---

## Architecture

```
Input Image (RGB, 384 × 384)
       │
       ▼
  TrOCR Processor
  (resize · normalize · patch embedding)
       │
       ▼
  ViT Encoder (Vision Transformer)
  Pretrained on handwritten document images
  → Patch-level visual features
       │
       ▼
  GPT-2 Decoder (auto-regressive)
  Custom Devanagari character-level vocabulary
  Smart embedding initialization from English weights
  → Character sequence predictions
       │
       ▼
  Greedy / Beam Search Decoding
```

**Model:** `microsoft/trocr-small-handwritten`
**Parameters:** 28,877,824 total
**Base architecture:** ViT (encoder) + GPT-2 (decoder)

---

## Custom Devanagari Tokenizer

Unlike the CRNN's fixed CTC vocabulary, TrOCR uses an auto-regressive decoder requiring a proper tokenizer:

| Token | ID | Description |
|-------|------|-------------|
| `<pad>` | 0 | Padding token |
| `<bos>` | 1 | Beginning of sequence |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown character |
| Devanagari chars | 4+ | One-to-one character mapping |

**Smart Embedding Initialization:** English pretrained embeddings are projected into the Devanagari token space rather than using random initialization, enabling faster convergence during fine-tuning.

---

## Training Strategy

### Two-Phase Approach

| Phase | Epochs | Strategy | Learning Rate |
|-------|--------|----------|---------------|
| **Phase A** | 3 | Freeze encoder, train decoder only | 5e-5 (decoder) |
| **Phase B** | 15 | Unfreeze all, differential LR | 1e-5 (encoder), 5e-5 (decoder) |

**Rationale:**
- Phase A quickly adapts the decoder to Devanagari characters while preserving the encoder's pretrained visual features.
- Phase B fine-tunes the full model with a lower encoder LR to prevent catastrophic forgetting of handwriting features.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Total Epochs** | 18 (3 + 15) |
| **Batch Size** | 16 |
| **Optimizer** | AdamW |
| **Scheduler** | Cosine (with warmup) |
| **Warmup Steps** | configurable |
| **Gradient Clipping** | 5.0 |
| **Mixed Precision** | ON (AMP + GradScaler) |
| **Loss** | Cross-Entropy (standard auto-regressive) |
| **SWA** | Enabled in last 3 Phase B epochs |
| **Platform** | Kaggle T4 GPU |
| **Training Time** | ~8–10 hours |

### Stochastic Weight Averaging (SWA)

- Activated during the final 3 epochs of Phase B (epochs 16–18).
- Model weights are averaged across checkpoints to improve generalization.
- SWA skip BN update since TrOCR uses LayerNorm exclusively.
- Final SWA val CER = 0.0207 — marginal improvement but not better than best checkpoint (0.0203), so original weights were retained.

---

## Dataset

| Split | Samples |
|-------|---------|
| Train | 150,000 |
| Val   | 20,000  |
| Test  | 30,000  |

Same 200k Hindi OCR dataset used in Phase 3 (Experiment 2 & 3), ensuring fair comparison between architectures.

---

## Image Preprocessing & Augmentation

**Pipeline:** Raw image → Grayscale → Autocontrast → Tight crop → RGB → Augmentation → TrOCR Processor

The preprocessing pipeline maintains consistency with Phase 2 and Phase 3, while the TrOCR processor handles the final resize (384×384) and normalization for the ViT encoder.

---

## Training Progress

### Phase A (Encoder Frozen — Epochs 1–3)

The decoder rapidly adapted to Devanagari characters. CER dropped quickly as the decoder learned the character-level mapping.

### Phase B (Full Fine-Tuning — Epochs 4–18)

| Epoch | Loss | Val CER | Val WER | LR | Notes |
|-------|------|---------|---------|----|-------|
| 14 | 0.0828 | 0.0225 | 0.0680 | 1.67e-06 | New best |
| 15 | 0.0754 | **0.0203** | **0.0655** | 9.62e-07 | **Overall best** |
| 16 | 0.0694 | 0.0204 | 0.0655 | 4.35e-07 | SWA started |
| 17 | 0.0658 | 0.0219 | 0.0670 | 1.10e-07 | SWA averaged |
| 18 | 0.0641 | 0.0214 | 0.0675 | 0.00e+00 | SWA finalized |

**Best Checkpoint:** Epoch 15 (Phase B), Val CER = **0.0203**

---

## Final Test Results

| Decoding Method | Test CER | Test WER | Char Accuracy | Word Accuracy | Time (30k samples) |
|----------------|----------|----------|---------------|---------------|---------------------|
| **Greedy** | 0.0531 | 0.1288 | 94.69% | 87.12% | 572s |
| **Beam (w=4)** | 0.0533 | 0.1290 | 94.67% | 87.10% | 1,296s |
| **Beam (w=8)** | **0.0534** | **0.1290** | **94.66%** | **87.10%** | 2,255s |

**Observation:** Beam search did not improve over greedy decoding for TrOCR. This is typical for auto-regressive decoder models where the greedy path is already high-confidence — unlike CTC-based models where beam search resolves blank/repeat ambiguities.

---

## Phase 3 CRNN vs Phase 3.5 TrOCR Comparison

| Metric | Phase 3 CRNN (Best) | Phase 3.5 TrOCR | Improvement |
|--------|---------------------|------------------|-------------|
| **Architecture** | ResNet-SE + BiLSTM(3×512) | ViT + GPT-2 | Transformer-based |
| **Parameters** | 22.4M | 28.9M | +29% |
| **Best Val CER** | 0.0413 | **0.0203** | −50.8% |
| **Test CER (Greedy)** | 0.0703 | **0.0531** | −24.5% |
| **Test WER (Greedy)** | 0.1577 | **0.1288** | −18.3% |
| **Test CER (Beam)** | 0.0696 | 0.0534 | −23.3% |
| **Test WER (Beam)** | 0.1561 | 0.1290 | −17.4% |
| **Char Accuracy** | ~93.0% | **94.7%** | +1.7pp |
| **Word Accuracy** | ~84.4% | **87.1%** | +2.7pp |

---

## Key Takeaways

1. **TrOCR significantly outperforms CRNN** — 24.5% relative CER reduction on the test set, validating the Transformer-based approach for Hindi HTR.

2. **Transfer learning is highly effective** — starting from pretrained English handwriting weights and fine-tuning on Devanagari delivers strong results with only 18 epochs of training.

3. **Two-phase training is critical** — freezing the encoder first lets the decoder adapt to the new character set before jointly fine-tuning prevents early divergence.

4. **Beam search offers minimal gains** — for TrOCR's auto-regressive decoder, greedy decoding is already near-optimal. This is a key difference from CTC-based models.

5. **SWA provides marginal improvement** — the SWA model (CER=0.0207) was slightly worse than the best checkpoint (CER=0.0203), suggesting the model had already converged well.

6. **Val-to-test gap is notable** — Val CER (0.0203) vs Test CER (0.0531) suggests some overfitting or domain shift. Future work could explore more aggressive regularization or domain adaptation.

---

## Output Artifacts

| File | Description |
|------|-------------|
| `best_model_trocr.pt` | Best TrOCR checkpoint (initial run, ~110 MB) |
| `best_model_trocr_final.pt` | Best TrOCR checkpoint (final run, ~110 MB) |
| `phase 3_5 output.ipynb` | Full training notebook (initial run) |
| `phase_3_5_output_final.ipynb` | Full training notebook (final run with improved results) |
| `phase_3_5_trocr_hindi_htr.ipynb` | Source notebook (code only, without outputs) |

---

## Next — Phase 4: Page-Level HTR

The word-level TrOCR model from Phase 3.5 becomes the recognition backbone for Phase 4's page-level system. Phase 4 adds **DBNet word detection**, **Needleman-Wunsch pseudo-label alignment**, and **LoRA fine-tuning** to extend from word-level to full handwritten page transcription.
