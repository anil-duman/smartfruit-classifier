# SmartFruit: Intelligent Fruit Classification System

## Overview

SmartFruit is an advanced fruit classification system that leverages deep learning to automatically identify and classify different fruit types. The system addresses key challenges in agricultural automation, including visual similarities between fruit classes (like mangoes and apples) and model generalization across varying environmental conditions.

### Key Challenges Addressed
- **Visual Similarity**: Distinguishing between visually similar fruits (mangoes vs apples)
- **Environmental Variations**: Robust performance under different lighting, angles, and backgrounds
- **Computational Efficiency**: Lightweight model suitable for real-world applications

## Features

- **High Accuracy**: Achieves 85% classification accuracy on test dataset
- **Transfer Learning**: Utilizes pre-trained MobileNetV2 for efficient feature extraction
- **Lightweight Architecture**: Optimized for resource-constrained environments
- **Balanced Performance**: Consistent precision, recall, and F1-score across all fruit classes
- **Real-time Processing**: Suitable for industrial and agricultural applications
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## Dataset

The dataset consists of **10,000 high-quality fruit images** across 5 categories:

| Category | Training | Validation | Testing | Total |
|----------|----------|------------|---------|-------|
| 🍎 Apple | 1,940 | 40 | 20 | 2,000 |
| 🍌 Banana | 1,940 | 40 | 20 | 2,000 |
| 🍇 Grape | 1,940 | 40 | 20 | 2,000 |
| 🥭 Mango | 1,940 | 40 | 20 | 2,000 |
| 🍓 Strawberry | 1,940 | 40 | 20 | 2,000 |
| **Total** | **9,700** | **200** | **100** | **10,000** |

### Dataset 
- **Source**: [Kaggle - Fruits Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification)

## Model Architecture

The SmartFruit system employs a **MobileNetV2-based architecture** with custom classification layers:

```
Input (224×224×3) → MobileNetV2 Backbone → Global Average Pooling → Dense Layer (Softmax) → Output (5 classes)
```

### Architecture Details
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen pre-trained layers + custom classification head
- **Activation**: Softmax for multi-class classification
- **Optimization**: Adam optimizer with categorical cross-entropy loss

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Batch Size | 128 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Image Size | 224×224 |
| Momentum | 0.9 |

## Results

### Overall Performance
- **Accuracy**: **85%** on test dataset
- **Training Efficiency**: Consistent improvement across 10 epochs
- **Generalization**: Strong performance on validation data

### Training Progress
| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|-------------------|---------------|----------------|
| 1 | 64.79% | 75.60% | 0.9271 | 0.6596 |
| 5 | 85.09% | 82.35% | 0.4246 | 0.4694 |
| 10 | **88.45%** | **83.65%** | **0.3383** | **0.4457** |

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Pandas
- OpenCV

## Performance Metrics

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 🍎 Apple | 0.80 | 0.80 | 0.80 | 20 |
| 🍌 Banana | 0.90 | 0.95 | **0.93** | 20 |
| 🍇 Grape | 0.86 | **0.95** | 0.90 | 20 |
| 🥭 Mango | 0.80 | 0.80 | 0.80 | 20 |
| 🍓 Strawberry | **0.88** | 0.75 | 0.81 | 20 |

### Overall Metrics
- **Macro Average**: Precision: 0.85, Recall: 0.85, F1-Score: 0.85
- **Weighted Average**: Precision: 0.85, Recall: 0.85, F1-Score: 0.85
- **Overall Accuracy**: **85%**

## Comparison with Other Methods

| Method | Dataset | Accuracy | F1-Score | Precision | Recall |
|--------|---------|----------|----------|-----------|--------|
| **SmartFruit (Ours)** | 10,000 images, 5 classes | **85%** | **0.85** | **0.85** | **0.85** |
| ResNet50 | FIDS-30 dataset | 86% | 0.84 | 0.93 | 0.89 |
| VGG16 | FIDS-30 dataset | 85% | 0.88 | 0.89 | 0.89 |
| MLP | Date fruit dataset | 92% | 0.90 | 0.91 | 0.90 |
| ConFruit CNN | 1,650 images, 3 classes | 61% | 0.69 | 0.71 | 0.68 |

### Advantages of SmartFruit
- Lightweight and efficient (MobileNetV2-based)
- Balanced performance across all classes
- No data augmentation required
- Fast training convergence (10 epochs)
- Suitable for resource-constrained environments

## Team Members

- Oğuzhan Öztürk
- Ali Özay
- Anil Duman
- Gizem Öztürk
- Berke Çevik
