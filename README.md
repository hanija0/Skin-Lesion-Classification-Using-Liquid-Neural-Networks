
# Skin Lesion Classification Using Liquid Neural Networks

## Title

**Liquid Neural Network-Based Stacking for Skin Lesion Classification**

---

# Overview

This repository implements a **robust, low-latency, and scalable skin lesion classification framework** designed to address critical challenges in automated dermatological AI deployment, including:

* Generalization across heterogeneous clinical datasets
* Class imbalance and broad diagnostic spectrums
* Overfitting in high-dimensional feature spaces
* Inference efficiency for real-time edge processing
* Model interpretability and causal stability

The system integrates complementary deep feature extractors through **adaptive continuous-time stacking via Liquid Neural Networks (LNNs)** to deliver stable and highly accurate diagnostic predictions without relying on explicit image segmentation.

The implementation corresponds to the research work:

**“Liquid Neural Networks based Stacking for Skin Lesion Classification.”**

---

# High-Level Architecture

<img width="610" height="210" alt="image" src="https://github.com/user-attachments/assets/daf9d049-799c-4b52-b3b9-0cabe66da638" />

The framework consists of three primary components:

1. **Quantum-Regularized YOLOv11x Backbone:** Captures rich global spatial context and features via Hilbert-space quantum-inspired variational regularization.
2. **MobileNetV3-Large Backbone:** Serves as a lightweight, highly efficient feature extractor for localized feature representations.
3. **Continuous-Time Liquid Neural Network (LNN) Meta-Classifier:** Dynamically processes concatenated out-of-fold (OOF) latent activations using ordinary differential equations (ODEs) to adjust integration weights based on feature complexity.

---

# Description

Existing deep learning systems for skin lesion classification suffer from significant clinical limitations:

* **Explicit Segmentation Failure:** Traditional pipelines require manual or standalone segmentation steps that remove surrounding healthy tissue context and propagate boundary errors.
* **Static Feature Aggregation:** Standard voting schemes, weighted averages, or static meta-learners (e.g., Logistic Regression) fail to capture non-linear logit interactions.
* **High Diagnostic Costs:** Early-stage detection (e.g., Stage I melanoma) significantly improves survival rates (nearly 94%), but diagnostic costs (~$150/visit) limit broad access.

This project resolves these challenges by using **implicit segmentation via object-detection backbones**, **quantum-inspired variational regularization**, **3-fold out-of-fold feature generation**, and **adaptive LNN meta-stacking** to ensure robust diagnostic performance across wide clinical lesion spectrums.

---

# Dataset Information

The framework is evaluated across two distinct clinical benchmarks:

| Dataset | Classes | Total Images | Train / Val / Test Split | Cross-Validation Scheme | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom Dataset** | 17 Classes | 3,036 | 2,121 / 455 / 460 (70:15:15) | **3-Fold Stratified CV** | Merged Derm12345 & Kaggle archives; Imbalance Ratio = 1.55; Normalized Entropy = 0.99. |
| **HAM10000** | 7 Classes | 10,015 | 7,007 / 1,502 / 1,506 (70:15:15) | **3-Fold Stratified CV** | Standard dermatoscopic benchmark; includes uncleaned raw clinical noise. |

* **Custom Dataset Link:** [Zenodo Download](https://zenodo.org/records/18896907)
* **HAM10000 Reference:** Tschandl P, Rosendahl C, Kittler H (2018). *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data 5:180161.

---

# Code Information

The repository contains two main execution notebooks:

```text
skin-lesion-classification/
│
├── custom.ipynb
├── ham10000.ipynb
└── README.md

```

### custom.ipynb

Executes the full 3-fold cross-validation training, feature extraction, quantum regularization, LNN stacking, and causal validation pipeline on the 17-class Custom dataset.

### ham10000.ipynb

Executes the comparative 3-fold pipeline and statistical benchmarking on the 7-class HAM10000 dataset.

---

# Requirements

Dependencies required to run the pipeline:

```text
Python >= 3.9
PyTorch
TensorFlow
NumPy
Pandas
Scikit-learn
Matplotlib
OpenCV
Ultralytics (for YOLOv11x)
Qiskit (for Quantum Variational Layer)

```

Install packages via `pip`:

```bash
pip install torch torchvision tensorflow ultralytics qiskit scikit-learn opencv-python pandas matplotlib

```

---

# Usage Instructions

### Step 1 — Clone Repository

```bash
git clone https://github.com/hanija0/skin-lesion-classification.git
cd skin-lesion-classification

```

### Step 2 — Download Dataset

Download the custom dataset from [Zenodo](https://zenodo.org/records/18896907) and extract it to:

```text
dataset/

```

### Step 3 — Run Training & Stacking

Run the desired notebook environment:

* For Custom Dataset (17 Classes): Execute `custom.ipynb`
* For HAM10000 (7 Classes): Execute `ham10000.ipynb`

---

# Methodology

1. **3-Fold Stratified Cross-Validation & Base Feature Extraction:**
* Training sets are split using **3-Fold Stratified CV** to generate unbiased Out-of-Fold (OOF) latent activations and prevent target leakage.
* **YOLOv11x:** Extracts 1280-dimensional global spatial embeddings.
* **MobileNetV3-Large:** Extracts 960-dimensional latent features from the average pooling layer.


2. **Quantum-Inspired Regularization:**
* YOLOv11x features (reduced to 256 dimensions) pass through a 4-qubit variational circuit (`ZZFeatureMap` + `RealAmplitudes` ansatz).
* Regularized loss via expectation variance:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \text{Var}(q(z))$$




3. **Meta-Feature Stacking:** 3-fold out-of-fold (OOF) feature vectors from both backbones are concatenated into a **1,224-dimensional** joint input representation.
4. **Liquid Neural Network Head:** An `OptimizedLiquidCell` processes the stacked vector via a continuous-time ODE over 10 simulation time steps ($\Delta t = 0.03$):

$$h_{t+1} = h_t + \Delta t \left( \tanh(W_{\text{state}} x) - h_t \cdot W_{\text{leak}} \right)$$


5. **Causal AI Analysis:** Interventional tests evaluate prediction stability under model feature perturbations using do-calculus ($E[Y \mid \text{do}(X \leftarrow X + 0.10)] - E[Y \mid X]$).

---

# Performance Summary

### Comprehensive Benchmark Evaluation

| Dataset | Model / Method | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Custom (17 Classes)** | ResNet50 | 0.8348 | 0.8428 | 0.8421 | 0.8364 | 0.9887 | 7.2958 |
|  | EfficientNet-B0 | 0.8413 | 0.8462 | 0.8471 | 0.8436 | 0.9911 | 9.1860 |
|  | Soft Voting Ensemble | 0.8043 | 0.8174 | 0.8108 | 0.8061 | 0.9893 | 0.0250 |
|  | Logistic Regression Stacking | 0.8517 | 0.8548 | 0.8582 | 0.8536 | 0.9735 | 0.0256 |
|  | **LNN Stacking (Proposed)** | **0.8609** | **0.8661** | **0.8647** | **0.8613** | **0.9920** | **0.0037** |
| **HAM10000 (7 Classes)** | MobileNetV3-Large | 0.8758 | 0.8240 | 0.7174 | 0.7526 | 0.9778 | 7.1104 |
|  | MLP Stacking | 0.8805 | 0.8625 | 0.7279 | 0.7725 | 0.9813 | 0.0231 |
|  | **LNN Stacking (Proposed)** | **0.8811** | **0.8398** | **0.7467** | **0.7789** | 0.9569 | **0.0152** |

### Statistical Validation

* **Friedman Test:** $\chi^2 = 20.00, p < 0.001$, confirming significant performance differences across methodologies.
* **Wilcoxon & Cliff's Delta:** Pairwise tests ($p = 0.002, \delta = 1.00$) show LNN Stacking achieves a statistically significant improvement over standalone backbones.

---

# Citations

If you use this project or framework in your research, please cite:

```bibtex
@article{edupuganti2026liquid,
  title={Liquid Neural Network-Based Stacking for Skin Lesion Classification},
  author={Edupuganti, Hanija and Vinta, Surender Reddy and Vege, Lahari and Ponnada, Dinesh Kumar},
  journal={School of Computer Science and Engineering, VIT-AP University},
  year={2026}
}

```

Dataset citation:

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific Data},
  volume={5},
  pages={180161},
  year={2018},
  publisher={Nature Publishing Group}
}

```

---

# License

This project is intended for **research and educational purposes only**. It is not cleared for clinical diagnostic use.

---

# Authors

**Hanija Edupuganti**, **Dr. Surender Reddy Vinta**, **Lahari Vege**, **Dinesh Kumar Ponnada**

*School of Computer Science and Engineering, VIT-AP University*
