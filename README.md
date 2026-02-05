# Skin-Lesion-Classification-Using-Liquid-Neural-Networks
Leveraged Liquid Neural Networks for skin lesion classification

## Overview

This project implements a **robust, scalable skin lesion classification system** designed to address real-world challenges in medical AI deployment, including **generalization, overfitting, inference efficiency, and model interpretability**.

The system integrates:

* A **quantum-regularized YOLOv11x** model
* A **lightweight MobileNetV3** classifier
* A **Liquid Neural Network (LNN)** meta-classifier for adaptive stacking

The primary focus of this work is **end-to-end inference pipeline design**, **model orchestration**, and **robust ensemble learning**, rather than standalone model accuracy.

The implementation corresponds to the accompanying research paper:

> *“Liquid Neural Networks based Stacking for Skin Lesion Classification”*

---

## Problem Motivation

Skin lesion classification systems face several deployment-critical limitations:

* Poor generalization due to limited and homogeneous datasets
* High sensitivity to class imbalance
* Overfitting in high-capacity models
* Dependency on explicit segmentation pipelines
* High inference cost in ensemble methods

This project addresses these limitations by:

* Eliminating explicit segmentation
* Using complementary backbone models
* Applying quantum-inspired regularization
* Employing Liquid Neural Networks for adaptive stacking

---

## High-Level Architecture

<img width="610" height="210" alt="image" src="https://github.com/user-attachments/assets/daf9d049-799c-4b52-b3b9-0cabe66da638" />

## Key Design Choices

### 1. No Explicit Segmentation

Unlike traditional pipelines, this system relies on **implicit segmentation capabilities** of modern architectures (YOLOv11x, MobileNetV3), avoiding:

* Error propagation from segmentation
* Additional preprocessing cost
* Loss of contextual background features

This improves scalability and real-world robustness.

---

### 2. Quantum-Inspired Regularization

To mitigate overfitting in YOLOv11x:

* A **4-qubit quantum circuit** is used as a regularization module
* Classical features are embedded into a Hilbert space
* A variance-based penalty is added to the loss function

This provides **stronger regularization than L2**, particularly in high-dimensional feature spaces.

---

### 3. Complementary Backbone Models

* **YOLOv11x**: Rich feature extraction, strong local and global representations
* **MobileNetV3-Large**: Lightweight, efficient, optimized for speed

Using both enables a **balanced accuracy–efficiency tradeoff**.

---

### 4. Liquid Neural Network (LNN) Stacking

Instead of static ensemble methods (soft voting, hard voting):

* Logits from base models are concatenated
* An **LNN meta-classifier** dynamically adapts its internal state
* Captures temporal dependencies in static feature activations

This results in:

* Better robustness
* Lower sensitivity to individual model perturbations
* Improved generalization across datasets

---

### 5. Causal & Sensitivity Analysis

To ensure reliability:

* Causal sensitivity tests quantify each model’s contribution
* Results show **balanced influence**, with no single model dominating
* Confirms stability under perturbations

---

## Dataset

* **Custom Dataset**: 17 skin lesion classes (balanced distribution). Available at: https://drive.google.com/file/d/1cXrsjCl5cI8W92VJIeaJt8_gPIPHnraI/view?usp=sharing
* **Benchmark Dataset**: HAM10000
* Split ratio: **70% Train / 15% Validation / 15% Test**
* No hair or noise removal to preserve real-world characteristics

---

## Performance Summary

* **Custom Dataset**

  * Accuracy: **88.7%%**
  * ROC-AUC: **0.987**
  * Inference time: **~0.17 s**

* **HAM10000**

  * Accuracy: **~89%**
  * Strong cross-dataset generalization

The LNN-based stacking consistently outperforms:

* Soft voting
* Hard voting
* Weighted averaging
* Logistic regression stacking

---

## Engineering & System Focus

This project emphasizes:

* End-to-end inference pipeline design
* Modular model orchestration
* Accuracy–latency tradeoff analysis
* Robust ensemble behavior
* Interpretability through causal analysis

---

## Project Structure

```
skin-lesion-classification/
├── custom.ipynb
├── ham10000.ipynb
└── README.md
```

---

## Limitations

* Clinical validation is outside the scope of this work
* Designed for decision-support, not diagnosis

---



## Author

**Hanija Edupuganti, Dr. Surender Reddy Vinta, Lahari Vege, Dinesh Kumar Ponnada**
School of Computer Science and Engineering
VIT-AP University

