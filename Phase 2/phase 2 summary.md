# Phase 2: Hindi HTR — Data Preprocessing & Pipeline Construction

## 📌 Overview
Phase 2 focuses on building a robust, production-ready data preprocessing pipeline for the Hindi Handwritten Text Recognition system. This phase was executed in **two iterations** — first with a smaller 92K-sample dataset, then upgraded to a larger 200K-sample dataset after the initial version did not yield sufficient performance for downstream model training (Phase 3).

Both preprocessing notebooks (`phase 2 output_92kdataset.ipynb` and `phase 2 output 200kdataset.ipynb`) produce standardized artifacts (character vocabulary, preprocessing config, PyTorch DataLoaders) consumed by all subsequent phases.

---

## 🔄 Dataset Evolution

### Iteration 1: 92K Dataset (IIIT-INDIC-HW-WORDS)

**Source:** [c3rl/IIIT-INDIC-HW-WORDS-Hindi](https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi) (HuggingFace)

| Split | Samples |
|-------|--------:|
| Train | 69,853 |
| Val   | 12,708 |
| Test  | 12,869 |
| **Total** | **95,430** |

- **Format:** 3 Parquet files for training (`train data 1/2/3.parquet`, ~440 MB each), 1 for validation (~246 MB), 1 for test (~243 MB)
- **Columns:** `image` (raw bytes), `text` (Hindi word label)
- **Vocabulary:** 103 tokens (100 Devanagari characters + `<BLANK>`, `<UNK>`, `<PAD>`)
- **Limitation:** The 69K training samples proved insufficient for the CRNN model (Phase 3) to generalize well on complex conjuncts and rare matras. Performance plateaued below target thresholds, prompting the dataset upgrade.

### Iteration 2: 200K Dataset (IIIT Hyderabad Hindi OCR) ✅

**Source:** [IIIT Hyderabad 200K Word Hindi Dataset](https://www.kaggle.com/datasets/nehamalik10/hindi-ocr-new-dataset) (Kaggle), converted to Parquet format

| Split | Samples |
|-------|--------:|
| Train | 150,000 |
| Val   | 20,000 |
| Test  | 30,000 |
| **Total** | **200,000** |

- **Format:** Single Parquet files — `train.parquet` (~1.9 GB), `val.parquet` (~236 MB), `test.parquet` (~421 MB)
- **Columns:** `image_name`, `image` (raw bytes), `text` (Hindi word label)
- **Vocabulary:** 139 tokens (136 Devanagari characters + `<BLANK>`, `<UNK>`, `<PAD>`)
- **Improvement:** 36 additional characters captured (rare conjuncts, vowel modifiers like ऐ, ऑ, etc.), 2.15× more training data, and better coverage of the Devanagari Unicode range

---

## 🛠️ Preprocessing Pipeline (Common to Both Iterations)

The preprocessing pipeline is identical across both iterations, ensuring consistent data flow into Phase 3/3.5:

### Image Preprocessing
1. **Grayscale Conversion** — Convert RGB handwriting images to single-channel grayscale
2. **Otsu Binarization** — Adaptive thresholding to separate ink from background (togglable via `skip_binarization`)
3. **Tight Cropping** — Remove surrounding whitespace with 2px padding to focus on the handwritten content
4. **Aspect-Ratio-Preserving Resize** — Scale to target height of **32px** while maintaining aspect ratio, then pad to fixed width of **128px**
5. **Normalization** — Pixel values mapped from `[0, 255]` → `[-1, 1]` via `(x/255 - 0.5) / 0.5`

### Data Augmentation (Training Only)
| Augmentation | Parameters |
|-------------|-----------|
| Rotation | ±5° random |
| Elastic Distortion | Grid-based warping |
| Gaussian Noise | Random intensity |
| Erode/Dilate | Morphological transforms for stroke variation |
| Shear/Slant | ±12° horizontal shear |
| Cutout/Coarse Dropout | Random rectangular regions masked |

### Label Processing
- **Unicode NFC Normalization** — All Hindi text labels normalized to NFC form for consistent character representation
- **Unknown Character Cleaning (200K only)** — Labels containing non-Devanagari characters (e.g., `O`, `P`, `[`, `\`, `e`, `i`, `|`) were cleaned:
  - Validation set: 93 out of 20,000 labels cleaned
  - Test set: 304 out of 30,000 labels cleaned
- **Character Vocabulary** — Built from training set, mapping each unique Devanagari character to a unique integer ID with special tokens for CTC decoding

---

## 📦 Artifacts Produced

The preprocessing pipeline generates artifacts on Kaggle at `/kaggle/working/phase2_preprocessing/`. Only the two essential JSON configuration files were downloaded locally for use in subsequent phases:

### Downloaded Locally (per iteration)
| Artifact | Description |
|----------|------------|
| `char_vocab.json` | Character-to-index and index-to-character mappings with special token IDs |
| `preprocess_config.json` | Full preprocessing configuration (image dimensions, augmentation params, split sizes) |

### Generated on Kaggle (visible in output notebooks)
The following artifacts and visualizations were generated during execution and are embedded in the output notebooks (`phase 2 output_92kdataset.ipynb` and `phase 2 output 200kdataset.ipynb`):

- `split_sizes.json` — Train/val/test sample counts
- `sample_train_batch.pt` — Sample PyTorch batch tensor for downstream sanity checks
- `image_dimension_sample.csv` — Raw image dimension statistics
- `unicode_character_analysis.csv` — Character frequency distribution
- **Visualizations:** Raw samples grid, preprocessing step-by-step views (4 pages), augmentation before/after comparison, dimension distribution plots, and character frequency histograms

---

## 🔧 PyTorch Integration

### Dataset Class: `HindiHTRDataset`
- Returns `(image_tensor, label_tensor, label_length)` tuples
- Image tensor shape: `[1, 32, 128]` (grayscale, 32×128 pixels)
- Labels encoded as integer sequences using the character vocabulary
- Augmentation applied only during training

### DataLoader: `htr_collate_fn`
- Custom collation for variable-length labels with padding
- Batch tensor shape: `[batch_size, 1, 32, 128]`
- Label tensors padded to max length in batch
- Batch size: 32


