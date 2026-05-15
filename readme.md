# Automatic License Plate Recognition (ALPR)
### Hybrid Image Processing & Machine Learning System

> Digital Image Processing — Spring Term 2026

---

## Overview

A two-track hybrid ALPR system that combines classical image processing with deep convolutional feature extraction. The system is built across two complementary notebooks, each targeting a distinct sub-problem of the full recognition pipeline.

| Notebook | Dataset | Focus |
|---|---|---|
| `ALPR_Segmentation_based` | Multi-country TSV (no bounding boxes) | Plate **localization** from scratch |
| `dip-based-car-plate-number-detection` | Kaggle Car Plate Detection (XML annotations) | Plate **enhancement, segmentation & OCR** |

---

## Pipeline Architecture

```
INPUT IMAGE (BGR)
        │
        ▼
┌─────────────────────────────────────────┐
│  1. Adaptive Preprocessing               │
│     Diagnostics → Gamma / CLAHE /        │
│     Contrast Stretch / Median Filter     │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  2. Orientation Correction               │
│     Canny → HoughLinesP → Affine Warp   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  3. Car Region Localization              │
│     Dominant Object → Line Extraction   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  4. Plate Candidate Generation           │
│     HSV Color Masking + Edge Fallback   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  5. Candidate Scoring (6-factor)         │
│     Background · Text · Components ·    │
│     Projections · Edge Density · Texture │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  6. OCR Preprocessing                    │
│     Bilateral → CLAHE → Sharpen →       │
│     Adaptive Threshold → Morph → Invert │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  7. Character Recognition                │
│     Tesseract PSM8/OEM3                 │
│     (+ Template Matching fallback)      │
└──────────────┬──────────────────────────┘
               ▼
        PLATE STRING OUTPUT

── Parallel Branch ──────────────────────────
img_rot → VGG19 / ResNet50 / EfficientNetB0
        + LBP / GLCM / SIFT / ORB
        → Concatenation (28,602-D)
        → StandardScaler → PCA (95% var)
        → SVM / RF / KNN / CNN Classifier
        → Quality / Difficulty / Authenticity Label
```

---

## Key Techniques

### Preprocessing
- **Adaptive diagnostics** — 14 per-image metrics mapped to 8 correction flags (`dark_image`, `blurry`, `noisy`, `overexposed_regions`, etc.)
- **Gamma correction** (γ = 1.5), contrast stretching (p2–p98), highlight compression
- **Median filtering** — 3×3, robust to salt-and-pepper noise
- **CLAHE** — tile-wise adaptive contrast (`clipLimit=2.0` for full image; `0.8` for plate crops)

### Frequency-Domain Filtering
- **2D FFT** with `np.fft.fft2` + `fftshift` for centered spectrum
- **Circular low-pass filter** (cutoff radius r₀ = 30) — suppresses noise and compression artifacts
- **High-boost filter** (β = 1.2, r₀ = 25) — sharpens character stroke edges

### Plate Localization (Notebook 1)
- Canny edge detection with border zeroing
- Hough Line Transform for rotation estimation (median angle across qualifying segments)
- Dominant object detection via contour area and aspect-ratio filtering
- **HSV color-space masking** for white `(S<75, V>95)` and yellow `(15≤H≤45)` plates
- IoU-based NMS (threshold 0.35) for candidate deduplication
- Geometric filters: aspect ratio ∈ [2.0, 6.0], area ratio ∈ [0.00025, 0.14]

### OCR Pipeline
- Bilateral denoising `(d=9, σ=75)` → CLAHE → unsharp-mask sharpening kernel
- Gaussian adaptive thresholding `(blockSize=11, C=2)`
- Morphological open/close cleanup + inversion
- **Tesseract** `--psm 8 --oem 3` as primary; MSE template matching as fallback

### Feature Extraction & Fusion

| Extractor | Dimensions | Notes |
|---|---|---|
| VGG19 | 25,088 | ImageNet pretrained, pool5 output |
| ResNet50 | 2,048 | Skip-connection backbone, avgpool output |
| EfficientNetB0 | 1,280 | Compound-scaled, SE attention |
| LBP (P=8, R=1, uniform) | 10 | Rotation-invariant texture histogram |
| GLCM (4 angles, 4 props) | 16 | Contrast, correlation, energy, homogeneity |
| SIFT (averaged) | 128 | Scale/rotation-invariant keypoints |
| ORB (averaged float) | 32 | Binary descriptor, fast inference |
| **Unified (pre-PCA)** | **28,602** | Full fused vector |
| **After PCA (95% var)** | **~9** | Compact, decorrelated |

---

## Datasets

### Dataset 1 — Multi-Country License Plate Dataset
- ~100 images, multiple countries (EU, Middle East, Asia)
- TSV metadata with plate text ground truth
- **No bounding box annotations** → full localization pipeline required
- Difficulty split: Easy 48 · Medium 46 · Hard 5 · Very Hard 1

### Dataset 2 — Kaggle Car Plate Detection (`andrewmvd/car-plate-detection`)
- 433 images with Pascal VOC XML bounding box annotations
- Enables direct plate cropping; OCR-focused experimentation
- No plate text ground truth

---

## Results

### Classifier Performance (PCA-reduced fused features)

| Classifier | Accuracy | F1-Score | ROC-AUC | Kappa |
|---|---|---|---|---|
| CNN (fine-tuned head) | **0.921** | **0.919** | **0.971** | **0.876** |
| SVM (RBF kernel) | 0.891 | 0.887 | 0.952 | 0.831 |
| Random Forest | 0.872 | 0.870 | 0.941 | 0.806 |
| KNN (k=5) | 0.834 | 0.831 | 0.903 | 0.753 |

### OCR Method Comparison

| Metric | Template Matching | Tesseract PSM8/OEM3 |
|---|---|---|
| Character rate (clean plates) | ~71% | **~82%** |
| Character rate (degraded plates) | ~34% | **~58%** |
| Font sensitivity | High | Low (LSTM-based) |
| Processing time / character | <1 ms | ~15 ms |

### CNN Backbone Comparison

| Architecture | Params | Feature Dims | Deployment Target |
|---|---|---|---|
| VGG19 | 143.7M | 25,088 | Cloud (offline) |
| ResNet50 | 25.6M | 2,048 | Server / Cloud ✓ |
| EfficientNetB0 | 5.3M | 1,280 | Edge / Mobile ✓ |

---

## Dependencies

```
opencv-python
numpy
scikit-learn
scikit-image
tensorflow / keras        # VGG19, ResNet50, EfficientNetB0
pytesseract               # OCR engine
Pillow
scipy
matplotlib
```

Install Tesseract separately:
```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

Install Python packages:
```bash
pip install opencv-python numpy scikit-learn scikit-image tensorflow pytesseract pillow scipy matplotlib
```

---

## Usage

### Notebook 1 — Segmentation-Based (unannotated data)
```python
# Run the full localization + OCR pipeline on a single image
results = process_single_image("path/to/car_image.jpg", tsv_metadata)

# Output: plate string, OCR confidence, top candidate bounding box
print(results["plate_text"])       # e.g. "ABC 1234"
print(results["ocr_confidence"])   # e.g. 0.73
print(results["best_box"])         # (x, y, w, h)
```

### Notebook 2 — Annotation-Guided (XML bounding boxes)
```python
# Parse XML annotation and extract plate crop
plate_crop = extract_plate_from_annotation("Cars0.png", "Cars0.xml")

# Run OCR preprocessing + recognition
binary   = preprocess_plate_v2(plate_crop)
chars    = segment_characters(binary)
plate_str = "".join(recognize_character_template_matching(c, templates) for c in chars)
```

### Feature Extraction + Classification
```python
features = extract_all_features(image_bgr)
# Returns 28,602-D unified vector

X_scaled = scaler.transform([features])
X_pca    = pca.transform(X_scaled)
label    = clf.predict(X_pca)   # SVM / RF / KNN / CNN
```

---

## Project Structure

```
.
├── ALPR_Segmentation_based.ipynb          # Notebook 1 — localization pipeline
├── dip-based-car-plate-number-detection.ipynb  # Notebook 2 — OCR pipeline
├── data/
│   ├── Car License Plate Detection Dataset.tsv
│   ├── images/                            # Dataset 1 images (by country)
│   ├── kaggle_images/                     # Dataset 2 PNG images
│   └── annotations/                       # Dataset 2 Pascal VOC XML files
├── outputs/
│   ├── preprocessed/
│   ├── candidates/
│   └── ocr_results/
└── README.md
```

---

## Known Limitations

- **Color-first localization** relies on HSV thresholds tuned for white/yellow plates; non-standard colors (blue, dark backgrounds) may not be detected.
- **Template matching** uses `FONT_HERSHEY_SIMPLEX` templates and is sensitive to real-world font variation.
- **No IoU-based localization evaluation** for Notebook 1 — Dataset 1 lacks bounding box ground truth.
- **VGG19 inference** (~0.5 s/image on CPU) makes real-time use impractical without GPU acceleration.
- **Manual median filter** in Notebook 2 (nested loops) is slow for large datasets; replace with vectorized NumPy for production.

---

## Future Work

- Replace classical localization with **YOLOv8** for real-time plate detection
- Replace Tesseract with a **CRNN + CTC** end-to-end sequence recognizer
- Evaluate **TrOCR / PaddleOCR** for multi-language support
- Quantize EfficientNetB0 to **INT8** (TensorRT / TFLite) for edge deployment
- Extend to **video streams** with DeepSORT tracking and multi-frame OCR voting

---

## References

Key references — see the full report for the complete list (22 citations).

- Du et al. (2013) — ALPR State-of-the-Art Review. *IEEE TCSVT*
- He et al. (2016) — ResNet. *CVPR*
- Tan & Le (2019) — EfficientNet. *ICML*
- Simonyan & Zisserman (2015) — VGGNet. *ICLR*
- Smith (2007) — Tesseract OCR Engine. *ICDAR*
- Gonzalez & Woods (2018) — *Digital Image Processing*, 4th ed.
