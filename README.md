# Diabetic Retinopathy Detection using Deep Learning

This project implements deep learning models to detect and classify diabetic retinopathy severity from retinal fundus images. The models can classify images into five categories: No Disease, Mild, Moderate, Severe, and Proliferate.

## Project Overview

Diabetic retinopathy is a diabetes complication that affects the blood vessels in the retina, potentially leading to blindness if not diagnosed early. This project aims to assist ophthalmologists in early diagnosis by automatically classifying the severity of the condition from retinal images.

## Models Implemented

1. EfficientNet
   - Architecture: EfficientNetB0 with custom top layers
   - Performance: 43% accuracy, 0.40 weighted avg F1-score
   - Best performing on "No disease" class (F1: 0.62)

2. ResNet
   - Architecture: ResNet50 with custom classification head
   - Performance: 49% accuracy, 0.42 weighted avg F1-score
   - Strong performance on "No disease" (F1: 0.70) and improved detection of Proliferate cases (F1: 0.37)

## Dataset

- Total images: 35,126 retinal fundus images
- Classes: 
  - No disease: 25,810
  - Mild: 2,443
  - Moderate: 5,292
  - Severe: 873
  - Proliferate: 708
- Dataset source: [Kaggle - Diabetic Retinopathy Resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)

## Project Structure

```
├── data_processing.ipynb    # Data preprocessing and augmentation
├── Classification_using_EfficientNet.ipynb      # EfficientNet model implementation
├── Classification_using_ResNet.ipynb            # ResNet model implementation
├── Dataset/
│   ├── resize_images_sample/   # Preprocessed images(Small Sample)
│   └── trainLabels.csv  # Image labels
```

## Requirements

- TensorFlow 2.x
- Keras
- OpenCV
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Data Preprocessing

1. Images are resized to 244x244 pixels
2. Class imbalance handled by random sampling of majority class
3. Data augmentation techniques:
   - Random rotations
   - Horizontal/vertical flips
   - Brightness adjustments
   - Contrast modifications
   - Random zoom

## Model Training

Both models were trained with:
- Batch size: 16-32
- Learning rate: 1e-5 (EfficientNet), 1e-3 (ResNet)
- Early stopping and learning rate reduction on plateau
- Class weights to handle imbalance
- Train/Validation/Test split: 70/15/15

## Results

### EfficientNet
- Overall accuracy: 43%
- Class-wise F1-scores:
  - No disease: 0.62
  - Mild: 0.07
  - Moderate: 0.20
  - Severe: 0.12
  - Proliferate: 0.04

### ResNet
- Overall accuracy: 49%
- Class-wise F1-scores:
  - No disease: 0.70
  - Mild: 0.16
  - Moderate: 0.00
  - Severe: 0.29
  - Proliferate: 0.37

## Challenges and Future Work

1. Limited computational resources (models were trained on local laptops) restricted:
   - Batch size optimization
   - Model architecture experimentation
   - Extended training periods
2. Class imbalance remains a significant challenge
3. Model performance on minority classes needs improvement
4. Future work could explore:
   - Ensemble methods
   - Advanced data augmentation techniques
   - Alternative architectures
   - Additional preprocessing steps
