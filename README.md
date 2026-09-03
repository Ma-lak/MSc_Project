# MSc_Project
# Deep Learning-Based Focal Position Prediction for Larval Zebrafish

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

This repository implements a deep-learning approach for predicting the axial focal position of larval zebrafish from microscopy images. The primary objective is to enable rapid focus estimation that can eventually drive real-time control of an **Electrically Tuneable Lens (ETL)** during live imaging of freely moving zebrafish.

---

## Overview

The central model leverages a modified **MobileNetV2** architecture trained on zebrafish Z-stack image patches. It predicts two distinct properties to establish focus:

1. **Focal Distance Magnitude** — Regression output.
2. **Focal Direction** — Classification output (`Negative` vs. `Positive`).

These outputs are combined to calculate a single **signed focal-distance prediction** ($\pm \mu\text{m}$).

---

## Model Architecture

The standard MobileNetV2 architecture has been customized to handle grayscale microscopy input:

* **Input:** Single-channel grayscale image patches ($1 \times H \times W$).
* **Outputs:** 
  * Focal-distance magnitude (continuous value).
  * Class logits for negative focal direction.
  * Class logits for positive focal direction.
* **Scaling:** The predicted magnitude is scaled directly by the Z-stack step spacing used during dataset acquisition.

---

## Dataset Structure

The dataset comprises cropped image patches extracted from Z-stacks with ground-truth focal offsets. Each record contains:

* `filename`: Path to the image patch.
* `focal_distance`: True offset distance ($\mu\text{m}$).
* `focal_direction`: Binary direction label (`0` for negative, `1` for positive).

> **Note:** To prevent data leakage and evaluate true generalization, completely unseen fish specimens are reserved exclusively for the test split.

---

## Training Setup

The network learns both regression and direction tasks simultaneously in a multi-task learning configuration:

* **Optimizer:** Adam
* **Loss Functions:**
  * $\mathcal{L}_{\text{regression}}$: Mean Squared Error (MSE)
  * $\mathcal{L}_{\text{classification}}$: Cross-Entropy Loss
* **Data Augmentation:** Random affine transformations (rotation, translation, scaling, shearing) to improve spatial invariance.

---

## Evaluation & Performance

Model performance is measured across both prediction heads and system speed:

* **Regression:** Mean Absolute Error (MAE in $\mu\text{m}$).
* **Classification:** Binary Direction Accuracy (%).
* **Latency:** Per-frame inference execution time (ms) to assess viability for closed-loop, real-time ETL control.