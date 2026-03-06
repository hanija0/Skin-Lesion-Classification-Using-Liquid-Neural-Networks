

# Skin Lesion Classification Using Liquid Neural Networks

## Title

**Skin Lesion Classification Using Liquid Neural Networks with Quantum-Regularized YOLOv11x and MobileNetV3**

---

# Overview

This repository implements a **robust and scalable skin lesion classification framework** designed to address key challenges in medical AI deployment, including:

* Generalization across datasets
* Class imbalance
* Overfitting in deep models
* Inference efficiency
* Model interpretability

The system integrates multiple models through **adaptive stacking using Liquid Neural Networks (LNNs)** to produce a stable and generalizable classification pipeline.

The implementation corresponds to the research work:

**“Liquid Neural Networks based Stacking for Skin Lesion Classification.”**

---

# High-Level Architecture
<img width="610" height="210" alt="image" src="https://github.com/user-attachments/assets/daf9d049-799c-4b52-b3b9-0cabe66da638" />


The framework consists of three main components:

1. **YOLOv11x Feature Extractor**
2. **MobileNetV3-Large Lightweight Classifier**
3. **Liquid Neural Network Meta-Classifier**

The predictions from base models are stacked and processed by the **LNN meta-classifier**, which dynamically adapts its internal state to improve classification robustness.

---

# Description

Skin lesion classification systems often struggle with:

* Dataset bias
* Overfitting
* Computational inefficiency
* Dependence on segmentation pipelines

This project addresses these issues by:

* Eliminating explicit segmentation
* Using complementary backbone architectures
* Introducing quantum-inspired regularization
* Applying adaptive stacking via Liquid Neural Networks

The goal is to build a **robust inference pipeline rather than focusing solely on single-model accuracy**.

---

# Dataset Information

This project uses two datasets.

### 1. Custom Skin Lesion Dataset

* **Classes:** 17 skin lesion categories
* **Distribution:** Balanced
* **Source:**

[https://drive.google.com/file/d/1cXrsjCl5cI8W92VJIeaJt8_gPIPHnraI/view?usp=sharing](https://drive.google.com/file/d/1cXrsjCl5cI8W92VJIeaJt8_gPIPHnraI/view?usp=sharing)

### 2. HAM10000 Dataset

A widely used dermatology benchmark dataset.

Dataset paper:

Tschandl P, Rosendahl C, Kittler H (2018).
The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific Data 5:180161.

### Data Split

Train : 70%
Validation : 15%
Test : 15%

No hair removal or artifact cleaning was performed to preserve **real-world imaging conditions**.

---

# Code Information

The repository contains two main notebooks.

```
skin-lesion-classification/
│
├── custom.ipynb
├── ham10000.ipynb
└── README.md
```

### custom.ipynb

Runs the complete pipeline using the **custom dataset**.

### ham10000.ipynb

Runs the complete pipeline using the **HAM10000 benchmark dataset**.

Both notebooks implement:

* Data preprocessing
* Base model training
* Logit extraction
* LNN stacking
* Evaluation metrics

---

# Requirements

Install the following dependencies.

```
Python >= 3.9
PyTorch
TensorFlow
NumPy
Pandas
Scikit-learn
Matplotlib
OpenCV
Ultralytics (for YOLO)
```

Install packages using:

```
pip install torch torchvision
pip install tensorflow
pip install ultralytics
pip install scikit-learn opencv-python pandas matplotlib
```

---

# Usage Instructions

### Step 1 — Clone Repository

```
git clone https://github.com/hanija0/skin-lesion-classification.git
cd skin-lesion-classification
```

### Step 2 — Download Dataset

Download the custom dataset from:

[https://drive.google.com/file/d/1cXrsjCl5cI8W92VJIeaJt8_gPIPHnraI/view?usp=sharing](https://drive.google.com/file/d/1cXrsjCl5cI8W92VJIeaJt8_gPIPHnraI/view?usp=sharing)

Place the dataset inside:

```
dataset/
```

---

### Step 3 — Run Training

For the custom dataset:

```
run custom.ipynb
```

For HAM10000:

```
run ham10000.ipynb
```

---

### Step 4 — Evaluation

The notebooks output:

* Accuracy
* ROC-AUC
* Confusion matrix
* Inference time
* Causal sensitivity scores

---

# Methodology

The proposed pipeline follows these steps.

### Step 1 — Feature Extraction

Two complementary models are used.

**YOLOv11x**

* Rich feature extraction
* Captures global spatial patterns

**MobileNetV3-Large**

* Lightweight architecture
* Efficient inference

---

### Step 2 — Quantum-Inspired Regularization

To mitigate overfitting in YOLOv11x:

* Classical features are mapped into a **4-qubit Hilbert space**
* Variance penalty is applied to the training loss
* Encourages smoother feature representations

---

### Step 3 — Feature Stacking

Outputs from both models are converted to **logit vectors** and concatenated.

```
Stacked feature vector = [YOLO logits | MobileNet logits]
```

---

### Step 4 — Liquid Neural Network Meta-Classifier

A **Liquid Neural Network (LNN)** processes the stacked features.

Advantages:

* Dynamic internal state
* Adaptive feature weighting
* Robust ensemble learning

---

### Step 5 — Causal Sensitivity Analysis

Model robustness is evaluated using:

* Perturbation tests
* Contribution scores
* Stability analysis

This ensures **balanced contribution across base models**.

---

# Performance Summary

### Custom Dataset

Accuracy: **88.7%**
ROC-AUC: **0.987**
Inference Time: **~0.17 seconds**

---

### HAM10000

Accuracy: **~89%**

The LNN stacking consistently outperforms:

* Soft voting
* Hard voting
* Weighted averaging
* Logistic regression stacking

---

# Citations

If you use this repository in research, please cite:

Hanija Edupuganti, Surender Reddy Vinta, Lahari Vege, Dinesh Kumar Ponnada.
**Liquid Neural Networks based Stacking for Skin Lesion Classification.**

HAM10000 Dataset:

Tschandl P, Rosendahl C, Kittler H (2018).
The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific Data 5:180161.

---

# License

This project is intended for **research and educational purposes only**.

It should **not be used for clinical diagnosis**.

---

# Contribution Guidelines

Contributions are welcome.

Possible improvements include:

* additional datasets
* improved stacking methods
* interpretability techniques
* model compression for deployment

Please open an **issue or pull request** for discussion.

---

# Authors

Hanija Edupuganti
Dr. Surender Reddy Vinta
Lahari Vege
Dinesh Kumar Ponnada

School of Computer Science and Engineering
VIT-AP University

