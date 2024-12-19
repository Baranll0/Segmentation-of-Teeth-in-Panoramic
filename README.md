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

This project implements a deep learning-based solution for segmenting individual teeth from panoramic X-ray images. It utilizes the Nested U-Net (U-Net++) architecture, known for its exceptional performance in semantic segmentation tasks. Key features of this project include:

- **Advanced Preprocessing**: Image resizing, normalization, and multi-class mask generation.
- **Sophisticated Loss Function**: A combination of CrossEntropyLoss and DiceLoss to ensure precise segmentation.
- **Post-processing Techniques**: Connected Component Analysis (CCA) to extract bounding boxes for individual teeth.
- **Robust Evaluation**: Metrics such as Dice Coefficient, Combined Loss, and Accuracy to assess model performance.

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
- **Classes**: 33 unique classes (one for each tooth) and background labeled as 0.

---

## Model Architecture

The Nested U-Net (U-Net++) model was chosen for its enhanced feature extraction and spatial preservation capabilities. Its architecture consists of:

1. **Encoder**: Extracts hierarchical features using convolutional layers.
2. **Bottleneck**: Captures high-level semantic information.
3. **Decoder**: Reconstructs the segmented output with skip connections for spatial accuracy.
4. **Loss Function**: A hybrid of CrossEntropyLoss and DiceLoss for optimal segmentation performance.

---

## Training Pipeline

### Training Details

- **Batch Size**: 2
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: CrossEntropyLoss + DiceLoss
- **Metrics**: Dice Coefficient, Accuracy
- **Epochs**: 200

### Training Outputs

- **Best Model**: `best.pt` (lowest validation loss)
- **Last Model**: `last.pt` (final epoch)

---

## Results and Performance

### Performance Metrics

- **Final Combined Loss**: 0.199
- **Dice Coefficient**: 0.872
- **Accuracy**: 91.3%

### Visualization of Results

#### Prediction Examples:





#### Training Convergence:

Graphs illustrating training and validation metrics demonstrate steady convergence towards the optimal solution.

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

- **Incorporation of Augmentation Techniques**: To enhance model robustness against diverse X-ray imaging conditions.
- **Integration with Clinical Software**: For real-time usage in dental clinics.
- **Refinement of Post-processing**: Improve bounding box extraction using advanced techniques.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation".
2. Zhou, Z., Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). "UNet++: A Nested U-Net Architecture for Medical Image Segmentation".
3. PyTorch Documentation: [https://pytorch.org](https://pytorch.org)

---

This README is designed to provide a comprehensive and professional overview of the project, emphasizing technical details and clarity.

