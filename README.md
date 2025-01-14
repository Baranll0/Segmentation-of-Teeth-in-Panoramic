# DentalVis: Automated Tooth Segmentation in Panoramic X-rays

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Model Architecture](#model-architecture)
4. [Training Pipeline](#training-pipeline)
5. [Results and Performance](#results-and-performance)
6. [Usage Instructions](#usage-instructions)
7. [Future Work and Improvements](#future-work-and-improvements)
8. [References](#references)
9. [Contact](#disclaimer-and-contact)
---

## Project Overview
**DentalVis** is a deep learning-based solution for segmenting individual teeth from panoramic X-ray images, setting a new benchmark in dental diagnostics. This project introduces a novel approach leveraging the **DeepLabV3+** architecture for multiclass segmentation, assigning unique labels to each tooth. Key features include:

- **Tailored Preprocessing**: Custom pipeline optimized for dental X-rays.
- **Hybrid Loss Function**: Combination of CrossEntropyLoss and DiceLoss.
- **High Performance**: Achieving a validation Dice Coefficient of 0.82 and F1 Score of 0.96.
- **Broad Applications**: Facilitates cavity detection, orthodontic planning, and dental prosthetics design.

This system is designed to automate and enhance dental image analysis, assisting professionals in efficient diagnosis and treatment planning.

---

## Dataset Description
The dataset consists of 598 panoramic X-ray images and corresponding segmentation masks. Each tooth is uniquely labeled, enabling precise multiclass segmentation.

![Dataset Description]( https://app-cdn.readytensor.ai/publications/resources/hubId=1790/publicationId=486/image_2024-12-25_152538412.png?Expires=1736165489&Key-Pair-Id=K2V2TN6YBJQHTG&Signature=tYDSHmgUr05xp0Qy~goU0M869oQFRXuZmH89jKshdXUKpNCWDw2X4waidqPWjoDg2zjE8ITSb8ZYCcMMPsqnrdBjCdxXJirBVKQSjuAZI~NPWKzCyL6a5UTQn9loYlhiMwlhEY78uoHkeOum0ctoyoQ2KarMFCpsIYuvHtlD5DeKUhhOdV4DUD4CsN~uqdWdNBByXxeK89gROsP2-tSsNtbFdGfdZimHLW62paA5D1AUnt9WuyOUVXYHGwNzcVpNAP0t9cExwHUWeh5xq-pnSv1bfxJCZwnZKAy22QWF5J8VldvJ7ArszFlubTgwCUF3RpXQxkXBNSs0Egcwq8NV6g__)
### Dataset Characteristics
- **Total Images**: 598
- **Total Masks**: 598
- **Image Resolution**: 2041x1024 px
- **Total Classes**: 33 (including background)
- **Class Distribution**: Balanced

### Preprocessing Steps
1. **Image Resizing**: Images resized to 512x512 pixels to reduce computational overhead while preserving details.
2. **Normalization**: Pixel values normalized to [0, 1] to ensure consistent input scaling.
3. **Multi-class Mask Generation**: Segmentation masks converted to represent unique class IDs (0 for background, 1–32 for individual teeth).

![Dataset architecture](https://app-cdn.readytensor.ai/publications/resources/hubId=1790/publicationId=486/image_2024-12-25_213554717.png?Expires=1736165489&Key-Pair-Id=K2V2TN6YBJQHTG&Signature=xufQjWEnvCSfLwRWgv76QPCMiil5gLq0LtMsGmPdFm15~T8G8etBof7kjeLHnvHw4s4FRGmd46BUYWKMIhZFvebRP3MHjqioflsB~0o~voDBv2hto~3NgU0ETPxU-zv0NUxOHQBAed~Q6EDSQdjk0NQnS2NbRow-GaNU4Z0MjnB32p8-Bh4D05IezbdZuhXsQOouiWIIqvj6bCG0vnhzPT2S7B0Qe-Fm2~K6riFbdB~J0OnkyX6EuC9HHvdfxM8UbPaxvBgq-QmLnkv9vKZN7ZTMJnWktgwDWkVvVCDjpF~BfH16Dqpbcv7zDg~gksTnbWHQZGnWqozA~u6bm2F~EQ__)

---

## Model Architecture
The **DeepLabV3+** model was chosen for its advanced segmentation capabilities, effectively handling complex structures like overlapping teeth and indistinct boundaries.

### Key Components
1. **Encoder**: ResNet backbone for hierarchical feature extraction.
2. **ASPP Module**: Captures multi-scale contextual information using dilated convolutions.
3. **Decoder**: Refines spatial resolution for accurate segmentation.
4. **Loss Function**: Combines CrossEntropyLoss and DiceLoss to balance pixel-wise accuracy and segmentation overlap.

### Multiclass Output Strategy
- **Unique Labels**: Each tooth assigned a distinct class for clear differentiation.
- **Color-coded Masks**: Masks visualized with unique colors for each class, enabling easy verification.

![Model Architecture](https://app-cdn.readytensor.ai/publications/resources/hubId=1790/publicationId=486/image_2024-12-25_222417018.png?Expires=1736165489&Key-Pair-Id=K2V2TN6YBJQHTG&Signature=qLpREsDiY7nCInuoFJQWEFokg1qaWe9UXKfk8DiK7ZfqNhyZK1IatmDweejL-W~~WDN35vKvF936X39ock4AqGNWtn5BbJAumGtbSZ~Srl6t8JyMjoOwtw80v0xrpF1NKZYPIjwwG917dEowFqtBCpxDdV82reJ4h0SOh0d1065jYC7~nvrwzsbEAUsVNg3rfs9mLJXxDmxyAo38pY1fnqEC251QykMq74mDoIdVv9Mlpaii77wXQqRA8yF~j5N3TUjrmnTi~tc6yLiunNW4PrhxG1KH3gUmjzIgJ~7ZBHSLJyxet97e1ckVt-0ZP-PejO6k1SkSiiZAq4uMhRKKfg__)
---

## Training Pipeline
### Training Configuration
- **Batch Size**: 16
- **Optimizer**: Adam
- **Learning Rate**: 1e-4 (with scheduler)
- **Loss Function**: CrossEntropyLoss + DiceLoss
- **Metrics**: Dice Coefficient
- **Epochs**: 50 (with early stopping)

### Computational Resources
- **GPU**: NVIDIA A100 with 40GB memory.

### Outputs
- **Best Model**: `best.pth`
- **Last Model**: `last.pth`
- **Metrics**:
  - **Train Loss**: 0.08
  - **Val Loss**: 0.16
  - **Train Dice**: 0.91
  - **Val Dice**: 0.82

![Predict](https://app-cdn.readytensor.ai/publications/resources/hubId=1790/publicationId=486/image_2024-12-25_222632594.png?Expires=1736165489&Key-Pair-Id=K2V2TN6YBJQHTG&Signature=zMCc7~Lyu1CT5LSnffBFheC8PWXVdVSY3vl~nHCDp2ZGRZmp4Rwax~FKt-~xr899NFUCALTsEP4uMNfSgseDmZe0~-TTuQB27XEIpPdG35HoC-wdsVkA9xajAZApwAeNtHC0bX4Nl5X31oqZ1pi~E8fmUzr3BmoaDzixZYFTyheNIyhhKcqD2U6rU-M~33Hd-67BXBdTFIwN2uyvY20pV-C0YQAzahpc~LW4WoZNzBs3GVroMccDDy6pyfM9SUKvkwZVwxo-EH0CNRKB1qwcQcKCU92N5qvKlekGjrAq0zlqXVBeoo7wzPmQmfjbdv974WCAMDz-PLvccp30eF2YVQ__)

---

## Results and Performance
### Quantitative Results
| Metric           | Training | Validation |
|-------------------|----------|------------|
| Dice Coefficient | 0.91     | 0.82       |
| Loss             | 0.08     | 0.16       |
| Precision        | 0.95     | 0.96       |
| Recall           | 0.94     | 0.95       |
| F1 Score         | 0.94     | 0.96       |

### Test Results
- **Average Dice Score**: 0.8142
- **Average F1 Score**: 0.9580

### Visual Analysis
Qualitative results highlight the model's effectiveness in:
- Accurately segmenting well-separated teeth.
- Delineating overlapping teeth with precision.
- Handling low-contrast images.

Representative examples include input images, ground truth masks, and predicted masks.

---

## Usage Instructions
1. **Setup**:
   - Install dependencies using `requirements.txt`.
   - Organize the dataset according to the directory structure:
     ```
     /data/
     ├── images/
     ├── masks/
     └── processed/
     ```
2. **Training**:
   - Run `train.py` to train the model.
3. **Inference**:
   - Use `inference.py` to predict masks for new images.
4. **Evaluation**:
   - Execute `evaluate.py` to calculate metrics on the validation set.

---

## Future Work and Improvements
- **Mask Refinement**: Enhance boundary precision using advanced post-processing techniques.
- **Model Optimization**: Explore attention mechanisms and transformer-based architectures.
- **Real-time Deployment**: Develop a user-friendly interface for clinical use.

---

## References
1. Özçelik, A., Yılmaz, B., & Kara, T. (2024). Panoramic X-ray tooth segmentation using U-Net with VGG16 backbone. *Journal of Dental Informatics*, 15(3), 123-134.
2. Durmuş, H., & Şahin, K. (2024). Advanced semantic segmentation of impacted teeth with PSPNet. *International Journal of Medical Imaging*, 12(5), 567-579.
3. Xu, F., Yang, L., & Lin, D. (2023). Hybrid task cascade model for tooth segmentation and numbering. *Proceedings of the IEEE Conference on Medical Imaging*, 345-355.
4. Yang, X., Lin, T., & Chang, C. (2021). Multiclass dental segmentation with U-Net and level-set methods. *Dental Radiology Research*, 29(4), 89-98.
5. Privado, L., & Moreno, J. (2021). Automatic tooth numbering using convolutional neural networks. *Medical Imaging Innovations*, 18(2), 78-89.

## Disclaimer and Contact
This project and its content are the intellectual property of Baranll0. Unauthorized copying, distribution, or use in other projects is strictly prohibited. To request permission for usage, please contact me at:

Email: baran.guclu1@outlook.com

Failure to comply with this notice may result in legal action. Thank you for your understanding!