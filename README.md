# Segmentation of Teeth in Panoramic X-rays

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Results and Performance](#results-and-performance)
6. [Usage Instructions](#usage-instructions)
7. [Future Work and Improvements](#future-work-and-improvements)
8. [References](#references)

---

## Project Overview

This project implements a deep learning-based solution for segmenting individual teeth from panoramic X-ray images. It utilizes the **DeepLabV3+** architecture, known for its exceptional performance in semantic segmentation tasks. Key features of this project include:

- **Advanced Preprocessing**: Image resizing, normalization, and multi-class mask generation.
- **Sophisticated Loss Function**: A combination of CrossEntropyLoss and DiceLoss to ensure precise segmentation.
- **Post-processing Techniques**: Multi-class segmentation with unique labels for each tooth.
- **Robust Evaluation**: Metrics such as Dice Coefficient and Validation Loss to assess model performance.

This system is designed to assist dentists and radiologists by automating the tooth segmentation process, paving the way for more efficient and accurate dental assessments.

---

## Dataset Description

The dataset consists of panoramic X-ray images and corresponding segmentation masks. Each tooth is labeled with a unique class ID, enabling multi-class segmentation.

**Directory Structure:**

```bash
/data/
├── images/                # Input X-ray images (JPG format)
├── masks/                 # Corresponding segmentation masks
└── processed/             # Preprocessed data (resized, converted)
```

- **Image Format**: JPG images resized to 512x512 pixels.
- **Classes**: Each tooth is assigned a unique class, with background labeled as 0.

---

## Model Architecture

The **DeepLabV3+** model was chosen for its advanced feature extraction capabilities and effectiveness in segmentation tasks. Its architecture consists of:

1. **Encoder**: Utilizes ResNet as a backbone for hierarchical feature extraction.
2. **ASPP Module**: Captures multi-scale contextual information.
3. **Decoder**: Ensures spatial accuracy and generates the final segmented output.
4. **Loss Function**: A hybrid of CrossEntropyLoss and DiceLoss for optimal segmentation performance.

---

## Training Pipeline

### Training Details

- **Batch Size**: 16
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: CrossEntropyLoss + DiceLoss
- **Metrics**: Dice Coefficient
- **Epochs**: 50
- **GPU Used**: NVIDIA A100

### Training Outputs

- **Best Model**: `best.pth`
  - **Train Loss**: 0.08
  - **Val Loss**: 0.16
  - **Train Dice**: 0.91
  - **Val Dice**: 0.82
- **Last Model**: `last.pth`
  - **Train Loss**: 0.03
  - **Val Loss**: 0.205
  - **Train Dice**: 0.96
  - **Val Dice**: 0.82

---

## Results and Performance

### Performance Metrics

- **Best Model**:
  - **Train Loss**: 0.08
  - **Val Loss**: 0.16
  - **Train Dice**: 0.91
  - **Val Dice**: 0.82

### Test Results

When tested on the test dataset:

- **Average Dice Score**: 0.8142
- **Average F1 Score**: 0.9580

These metrics confirm the model's robust performance and generalization ability.

### Visualization of Results

#### Example Output

![Segmentation Results](https://i.imgur.com/XmW596w.png)
![Segmentation Results2](https://i.imgur.com/5n3eirO.png)

#### Training Convergence

Graphs illustrating training and validation loss and Dice coefficient show steady convergence, highlighting the model's generalization.

---

## Usage Instructions

1. **Setup**:

   - Install dependencies using `requirements.txt`.
   - Prepare the dataset according to the specified directory structure.

2. **Training**:

   - Run `train.py` to train the model.

3. **Inference**:

   - Use `inference.py` to predict masks for new images.

4. **Evaluation**:

   - Execute `evaluate.py` to calculate performance metrics on the validation set.

---

## Future Work and Improvements

- **Refinement of Masks**: Improve boundary precision using post-processing techniques.
- **Model Optimization**: Experiment with different architectures and optimizers for better performance.
- **Deployment**: Build a user-friendly interface for real-time segmentation in clinical settings.

---

## References

1. Chen, L.-C., et al. (2018). "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs".
2. PyTorch Documentation: [https://pytorch.org](https://pytorch.org)
---
