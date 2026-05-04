# Phase 1: Hindi HTR — Data Exploration & Baseline

## 📌 Overview
This directory contains the initial exploratory data analysis (EDA) and benchmarking stage for the Hindi Handwritten Text Recognition (HTR) project. 

The primary notebook in this phase (`phase 1 output.ipynb`) focuses on loading the dataset, understanding the text and image schemas, and establishing a baseline performance using an off-the-shelf OCR model.

## 🛠️ What Was Done
* **Environment & Font Setup:** Configured `easyocr` and `editdistance`, and integrated the `NotoSansDevanagari` font to correctly render Hindi text in Matplotlib visualizations.
* **Dataset Loading & Inspection:** Loaded 95,430 handwritten Hindi word images stored in Parquet format, split across training (69k), validation (12k), and test (12k) sets.
* **Data Conversion:** Built helper functions to extract raw byte strings from the dataset and convert them into PIL images (RGB format) for visual inspection.
* **Baseline Evaluation:** Evaluated the raw dataset against the pre-trained `EasyOCR` model to calculate Exact Match Accuracy and Character Error Rate (CER).

## 📉 Baseline Performance: Why is Accuracy 0% ?
During the baseline evaluation, the Exact Match Accuracy resulted in 0% (and a very high Character Error Rate). This is an expected outcome at this stage for several key technical reasons:

1. **The Domain Gap (OCR vs. HTR):** EasyOCR is heavily optimized for *printed* text and digital documents. Handwritten Text Recognition (HTR) involves irregular stroke thicknesses, varying slants, and inconsistent spacing, which standard bounding-box OCR models struggle to process.
2. **Complexity of Handwritten Devanagari:** * **The Shirorekha:** The continuous top line connecting Hindi words is perfectly straight in print, but curves or breaks in handwriting, confusing standard segmentation.
    * **Matras & Conjuncts:** Vowel modifiers and half-letters (samyuktakshars) overlap heavily in handwriting, making character-level isolation nearly impossible for a printed-text model.
3. **Lack of Preprocessing:** The baseline was run on raw images with varying dimensions, illumination, and background noise. Without a robust preprocessing pipeline (resizing, padding, binarization), the text features cannot be effectively isolated.

## Dataset used 
https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi
