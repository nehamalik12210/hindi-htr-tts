# Phase 3 — CRNN Model Training for Hindi Handwritten Text Recognition

## Overview

Phase 3 focuses on training a **CRNN (Convolutional Recurrent Neural Network)** with CTC loss for word-level Hindi handwritten text recognition. Building on the Phase 2 preprocessing pipeline and vocabulary artifacts, this phase systematically explores dataset scale and architectural improvements across **three progressive experiments** — achieving a final test CER of **6.96%** (≈93% character accuracy).

---

## Architecture

### Baseline CRNN (92k & 200k Experiments)

```
Input Image (1 × 32 × 128)
       │
       ▼
  7-layer VGG-style CNN
  (64 → 128 → 256 → 256 → 512 → 512 → 512 channels)
  BatchNorm + ReLU after each conv
  MaxPool to collapse height to 1, preserve width = 32
  Dropout2d (0.3) in blocks 5 & 7
       │
       ▼
  Squeeze + Permute → (B, 32, 512)
       │
       ▼
  2-layer Bidirectional LSTM (hidden=256)
  → (B, 32, 512) output
       │
       ▼
  Fully Connected → (B, 32, num_classes)
       │
       ▼
  Log-Softmax → CTC Decoding
```

**Parameters:** ~8.2M trainable

### Improved CRNN (Final Experiment on 200k)

```
Input Image (1 × 32 × 128)
       │
       ▼
  ResNet Backbone with Squeeze-and-Excitation (SE) Blocks
  - Residual skip connections (solves vanishing gradients)
  - SE channel attention inside each block
  - 14 conv layers (vs 7 in baseline)
       │
       ▼
  3-layer Bidirectional LSTM (hidden=512)
  (up from 2-layer × 256 in baseline)
       │
       ▼
  Fully Connected → (B, T, num_classes)
       │
       ▼
  Log-Softmax → CTC Decoding (Greedy + Beam Search)
```

**Parameters:** ~22.4M trainable

---

## Experiments & Results

### Experiment 1: Baseline CRNN on 92k Dataset

| Detail | Value |
|--------|-------|
| **Dataset** | 92k samples (Train: 69,853 · Val: 12,708 · Test: 12,869) |
| **Vocabulary** | 103 classes |
| **Epochs** | 30 |
| **Batch Size** | 64 |
| **Optimizer** | Adam (LR=0.001, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5) |
| **Mixed Precision** | ON (AMP + GradScaler) |
| **Gradient Clipping** | 5.0 |
| **Best Epoch** | 26 |
| **Best Val CER** | 0.0819 |
| **Test CER** | **0.0778** |
| **Test WER** | **0.3001** |
| **Char Accuracy** | **92.22%** |
| **Word Accuracy** | **69.99%** |
| **Training Time** | ~359 minutes (~6 hours) |
| **Platform** | Kaggle T4 GPU |

**Key Observations:**
- Loss dropped steeply in the first 5 epochs (2.14 → 0.37), then steadily decreased.
- LR reduced from 0.001 → 0.0005 at epoch 17, and again to 0.00025 at epoch 26 — both triggered clear CER improvements.
- Epoch 9 showed a validation spike (CER=0.42), likely due to gradient instability in the LSTM — model recovered fully by epoch 11.

---

### Experiment 2: Baseline CRNN on 200k Dataset

| Detail | Value |
|--------|-------|
| **Dataset** | 200k samples (Train: 150,000 · Val: 20,000 · Test: 30,000) |
| **Vocabulary** | 139 classes (expanded to cover wider character set) |
| **Epochs** | 30 |
| **Batch Size** | 64 |
| **Optimizer** | Adam (LR=0.001, weight_decay=1e-4) |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Mixed Precision** | ON |
| **Gradient Clipping** | 5.0 |
| **Best Epoch** | 30 |
| **Best Val CER** | **0.0643** |
| **Training Time** | ~360 minutes (~6 hours) |
| **Platform** | Kaggle T4 GPU |

**Key Observations:**
- The 2.15× larger dataset yielded a **21.6% relative improvement** in val CER (0.0819 → 0.0643) with the same architecture.
- The model kept improving throughout all 30 epochs, suggesting it could benefit from more training.
- LR reduction from 0.001 → 0.0005 triggered at epoch 27 produced a sharp improvement (CER 0.0788 → 0.0672 → 0.0643).

---

### Experiment 3: Improved CRNN (ResNet-SE + BiLSTM) on 200k Dataset

| Detail | Value |
|--------|-------|
| **Dataset** | 200k samples (Train: 150,000 · Val: 20,000 · Test: 30,000) |
| **Vocabulary** | 139 classes |
| **Epochs** | 60 |
| **Batch Size** | 64 |
| **Optimizer** | Adam (LR=0.001) |
| **Scheduler** | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| **Warmup** | 3 epochs (linear 0.1→1.0) |
| **Label Smoothing** | 0.1 (CTC regularisation) |
| **Mixed Precision** | ON |
| **Gradient Clipping** | 5.0 |
| **Best Epoch** | 33 |
| **Best Val CER** | **0.0413** |
| **Test CER (Greedy)** | **0.0703** |
| **Test WER (Greedy)** | **0.1577** |
| **Test CER (Beam, w=10)** | **0.0696** |
| **Test WER (Beam, w=10)** | **0.1561** |
| **TTA CER (500 samples)** | **0.0536** |
| **TTA WER (500 samples)** | **0.1560** |
| **Training Time** | ~8–10 hours |
| **Platform** | Kaggle T4 GPU |

**Architecture Upgrades Over Baseline:**
- ResNet backbone with **Squeeze-and-Excitation (SE)** blocks replaces VGG-7
- 3-layer BiLSTM with 512 hidden units (up from 2 × 256)
- **Beam-search decoding** (beam width 10) for inference
- Cosine-annealing LR with linear warmup
- Enhanced augmentation: perspective transform + brightness/contrast jitter
- Label-smoothing regularisation for CTC
- Gradient accumulation support
- Optional WeightedRandomSampler for rare-character oversampling
- **Test-Time Augmentation (TTA)** evaluation

---

## Progressive Improvement Summary

| Experiment | Architecture | Dataset | Test CER | Char Accuracy | Relative Δ |
|------------|-------------|---------|----------|---------------|------------|
| 1. Baseline CRNN | VGG-7 + BiLSTM(2×256) | 92k | 0.0778 | 92.22% | — |
| 2. Baseline CRNN | VGG-7 + BiLSTM(2×256) | 200k | 0.0643* | ~93.6%* | −17.4% CER |
| 3. Improved CRNN | ResNet-SE + BiLSTM(3×512) | 200k | **0.0696** | **~93.0%** | −10.5% CER |

*\*Val CER only; full test evaluation not separately reported for Experiment 2.*

**Key Takeaways:**
1. **Data scale matters significantly** — doubling the dataset size improved CER by ~21% with the same model.
2. **Architecture improvements yield further gains** — the ResNet-SE backbone achieved the best val CER (0.0413), though the gap narrowed on the test set due to the model's increased capacity.
3. **Beam search provides modest improvement** — CER improved from 0.0703 (greedy) to 0.0696 (beam w=10), a 1% relative gain.
4. **TTA is effective** — averaging predictions across augmented views reduced test CER from 0.0703 to 0.0536 on the 500-sample subset.
5. **Hardest characters:** Nukta (़), short vowel matras (ु, ू), and rare consonants (थ, छ, ष, श) had the highest per-character error rates.

---

## Preprocessing Pipeline

Identical to Phase 2 — ensures train/test consistency:

1. **Grayscale conversion** → Autocontrast → Median denoise
2. **Deskew** (±8° max)
3. **Tight crop** (padding=2px)
4. **Otsu binarization** → Ensure black-on-white
5. **Aspect-preserving resize** to 32 × 128 (center-padded)
6. **Normalize** to [-1, +1]

**Training Augmentation:**
- Random rotation (±5°, p=0.50)
- Elastic distortion (p=0.30)
- Gaussian noise (p=0.35)
- Random erode/dilate (p=0.15)
- Random shear (±12°, p=0.40)
- Random cutout (p=0.20)
- *Improved model adds:* Perspective transform + Brightness/contrast jitter

---

## Output Artifacts

| File | Description |
|------|-------------|
| `92k dataset/best_model_92kdataset.pt` | Best CRNN checkpoint (92k dataset, ~94 MB) |
| `200k dataset/best_model 200k.pt` | Best CRNN checkpoint (200k dataset, ~94 MB) |
| `final improved model on 200k dataset/best_model_improved.pt` | Best improved CRNN checkpoint (200k, ~257 MB) |
| `92k dataset/phase 3 output_92kdata.ipynb` | Full training notebook for 92k experiment |
| `200k dataset/phase 3 output 200kdataset.ipynb` | Full training notebook for 200k experiment |
| `final improved model on 200k dataset/phase 3 improved output.ipynb` | Full training notebook for improved model |

---

## Next — Phase 3.5: TrOCR Fine-Tuning

The CRNN model achieves strong baseline performance (~93% char accuracy). Phase 3.5 explores a fundamentally different architecture — **TrOCR (Transformer-based OCR)** — which replaces the CNN+LSTM pipeline with a ViT encoder and GPT-2 decoder, aiming for further accuracy gains through transfer learning from pretrained vision-language models.
