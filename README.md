# RetinaVision-CNN: Diabetic Retinopathy Detection using Inception V3

## Project Overview

Diabetic Retinopathy (DR) is a common complication of diabetes that can lead to blindness if not detected early. This project applies a Convolutional Neural Network (CNN), specifically the **Inception V3 model**, to detect and classify the severity of diabetic retinopathy from retina images.

This project focuses on:
- Fine-tuning the Inception V3 model on the **APTOS 2019 Blindness Detection dataset**.
- Using various image preprocessing and data augmentation techniques to improve model accuracy.

## Features
- **Image Preprocessing**: Techniques such as Gaussian blur, Sobel edge detection, and contrast enhancement are applied to the dataset for better feature extraction.
- **Classification**: The model classifies retina images into 5 categories: No DR, Mild, Moderate, Severe, and Proliferative DR.
- **Transfer Learning**: Inception V3 is pre-trained on ImageNet and fine-tuned for DR detection.

## Dataset
The project uses the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset. The dataset consists of 3,662 retina images labeled into five categories:
- 0: No DR
- 1: Mild DR
- 2: Moderate DR
- 3: Severe DR
- 4: Proliferative DR

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RetinaVision-CNN.git
   cd RetinaVision-CNN
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data` directory:
   ```bash
   mkdir data
   # Place APTOS 2019 dataset here
   ```

4. Run the training script:
   ```bash
   python train.py
   ```

## Project Structure
```
RetinaVision-CNN/
│
├── data/                  # Directory to store the dataset
├── models/                # Saved models after training
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code for the project
│   ├── data_loader.py     # Script to load and preprocess data
│   ├── model.py           # Inception V3 model architecture and fine-tuning
│   └── train.py           # Training and evaluation script
├── results/               # Directory for storing results, metrics, and plots
├── README.md              # Project overview
└── requirements.txt       # Required Python packages
```

## Image Preprocessing Techniques

In this project, several image preprocessing techniques were applied to enhance model performance:

1. **Sobel Edge Detection**: Detects edges in the retina images by highlighting areas of rapid intensity change.
2. **Canny Edge Detection**: Another edge detection method used to reduce noise and false positives.
3. **Gaussian Blur**: Reduces noise by smoothing the image.
4. **Contrast Enhancement**: Increases the image contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).

These techniques were applied separately and combined in various experiments to observe their impact on model accuracy.

## Model Architecture

The **Inception V3** model, pre-trained on ImageNet, was fine-tuned for this task. Transfer learning allowed us to leverage pre-trained weights and adapt them for DR detection. Key steps:
- First, the convolutional layers were frozen, and only the last classification layer was fine-tuned.
- Later, additional layers were unfrozen and fine-tuned for further training to improve accuracy.

## Results

After training on the APTOS 2019 dataset, the model achieved the following:
- Best performance was observed when training on **raw images** with an accuracy of **79%**.
- **Hybrid Sobel augmentation** slightly improved predictions in the "Proliferative DR" and "Severe DR" classes.
- Further image preprocessing did not consistently improve overall accuracy, but some techniques performed better in specific cases with less data.

## Experiments

Seven experiments were conducted, applying different image transformations to the input data:
1. **Original Images**: No preprocessing.
2. **Sobel Edge Detection**: Applied to emphasize edges.
3. **Hybrid Sobel**: Combined Sobel-processed images with original ones.
4. **Canny Edge Detection**: Pre-processed images with Canny edge detection.
5. **Gaussian Blur**: Applied Gaussian blur to smooth the images.
6. **Contrast Enhancement**: Enhanced contrast with CLAHE.
7. **Gaussian Blur + Contrast**: Combined Gaussian blur with contrast enhancement.

## Conclusion

This project highlights the potential of CNNs, particularly **Inception V3**, in automating the detection of diabetic retinopathy. While some preprocessing techniques improved detection in underrepresented cases (e.g., Proliferative DR), the overall best performance came from training on raw images. Future work could involve using larger datasets, more data augmentation, and experimenting with other CNN architectures.

## Future Work
- **Larger Datasets**: Training on larger datasets could further improve accuracy and reduce overfitting.
- **More Advanced Preprocessing**: Incorporating additional image processing techniques and conducting a hyperparameter search could yield better results.
- **Model Optimization**: Exploring newer architectures, such as Inception V4 or EfficientNet, might lead to better accuracy with fewer computational resources.

## License
This project is licensed under the [GNU GPLv3 License](LICENSE).

## Acknowledgments
We would like to thank the **Vision and Cognitive Science** course instructors at the University of Padova for their support, and the creators of the **APTOS 2019 Blindness Detection** dataset.
