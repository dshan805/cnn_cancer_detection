# CNN Cancer Detection Kaggle Mini-Project

## 1. Project Overview

This project focuses on developing a **Convolutional Neural Network (CNN) based classification model** to identify the presence of **metastatic cancer** in small image patches from digital pathology slides. The dataset used is from the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data) Kaggle competition.

**Project Significance & Business Impact:**
Pathologists spend considerable time manually reviewing numerous tissue slides, a process prone to fatigue and potential errors. This AI model aims to **automate and assist the initial screening phase**, which can **significantly enhance the efficiency of pathologists' workflow** and potentially **accelerate cancer diagnosis, leading to earlier treatment and improved patient outcomes**. When deploying in a real clinical setting, a critical consideration would be balancing the costs of False Positives versus False Negatives based on specific business and medical requirements.

## 2. Data Source

* **Dataset:** [Kaggle - Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
* **Data Structure:**
    * `train.csv`: A CSV file containing image IDs and binary labels (1: cancer, 0: normal).
    * `train/` and `test/` folders: Contain approximately 220,000 and 57,000 image patches (`*.tif`) respectively.
* **Data Characteristics:**
    * Image size: 96x96 pixels (RGB channels).
    * **Class Imbalance:** The training dataset exhibits class imbalance, with normal samples (label 0) being approximately 1.47 times more prevalent than cancer samples (label 1) (Normal: 130,908, Cancer: 89,117).

## 3. Methodology

### 3.1. Data Preprocessing & Augmentation

* **Image Loading & Scaling:** Images are loaded using the PIL library, and pixel values are normalized to the 0-1 range.
* **`ImageDataGenerator` Utilization:** Keras's `ImageDataGenerator` is employed for efficient batch processing of large image datasets. It also applies various augmentation techniques to the training data to improve the model's generalization capabilities. Applied augmentation techniques include:
    * `rescale`: 1./255 (pixel value normalization)
    * `rotation_range`: 20 (up to 20-degree rotation)
    * `zoom_range`: 0.1 (10% zoom in/out)
    * `horizontal_flip`: True
    * `vertical_flip`: True
* **Train/Validation Split:** The training dataset is split into an 80% training set and a 20% validation set to monitor model performance during training.

### 3.2. Model Architecture

A sequential **Convolutional Neural Network (CNN)** model was constructed.

* **Input Layer:** 96x96x3 (height, width, RGB channels)
* **Convolutional Blocks:**
    * `Conv2D` (3x3 kernel, ReLU activation): Filters progressively increase (32, 64, 128) to extract image features.
    * `BatchNormalization`: Applied after each convolutional layer to enhance training stability and speed.
    * `MaxPooling2D` (2x2 pooling): Reduces feature map dimensions, decreasing computational complexity and highlighting prominent features.
* **Feature Flattening:** A `Flatten` layer converts the 2D feature maps from convolutional layers into a 1D vector.
* **Fully Connected (Dense) Layers:**
    * `Dense` (256 units, ReLU activation): Learns high-level patterns from the flattened features.
    * `Dropout` (0.5): A 50% dropout rate is applied to prevent overfitting.
    * `Dense` (1 unit, Sigmoid activation): The final output layer for binary classification, outputting the probability of cancer.

### 3.3. Model Training

* **Optimizer:** Adam optimizer was used.
* **Loss Function:** `binary_crossentropy` (suitable for binary classification).
* **Evaluation Metrics:** `accuracy` and **`AUC` (Area Under the Receiver Operating Characteristic Curve)** were used. AUC is particularly robust for evaluating overall classification performance in datasets with class imbalance.
* **Callbacks:** `EarlyStopping` (patience=5, restore_best_weights=True) was applied. This stops training early if the validation loss does not improve for 5 consecutive epochs and restores the model weights from the best performing epoch to prevent overfitting.

## 4. Project Results

The developed CNN model achieved the following performance on the validation dataset:

* **Validation Accuracy:** (Insert your validation accuracy here, e.g., `87.5%`)
* **Validation AUC:** (Insert your validation AUC here, e.g., `0.94`)

These results indicate that the model demonstrates meaningful performance in detecting metastatic cancer, with the AUC metric suggesting good discriminative power between positive and negative classes.

## 5. Limitations & Future Enhancements

While this mini-project demonstrates the potential of CNNs for cancer detection, further considerations and improvements are necessary for real-world deployment:

* **Advanced Class Imbalance Handling:** Although AUC was used, explicitly addressing the class imbalance (e.g., using `class_weight` in the loss function, oversampling techniques like SMOTE on features, or undersampling) could further improve the model's ability to learn from the minority class (cancer).
* **Transfer Learning:** Leveraging pre-trained CNN models (e.g., ResNet, VGG, Inception) on large datasets like ImageNet can significantly boost performance, especially with medical imaging datasets where acquiring vast amounts of labeled data can be challenging. This is a common and highly effective approach in medical image analysis.
* **Model Interpretability (Explainable AI, XAI):** In medical contexts, understanding *why* a model made a particular prediction is as crucial as the prediction itself. Future work could involve applying XAI techniques like Grad-CAM or LIME to visualize which parts of the image contribute most to the model's prediction.
* **Hyperparameter Optimization:** Systematic tuning of hyperparameters (e.g., learning rate, batch size, number of layers/filters, dropout rates) using methods like Grid Search, Random Search, or more advanced libraries (e.g., Keras Tuner, Optuna) could further optimize model performance.
* **Real-time Deployment & Scalability (MLOps Perspective):**
    * **Containerization:** Packaging the model and its dependencies into Docker containers ensures consistent environments and simplifies deployment.
    * **Cloud Services:** Utilizing cloud resources such as AWS EC2/S3/SageMaker Endpoints or Azure VMs/Blob Storage/Azure ML Endpoints for scalable data storage, high-performance GPU-accelerated training, and efficient inference serving.
    * **CI/CD & Monitoring:** Establishing a Git-based CI/CD pipeline for automated model updates and continuously monitoring deployed model performance (e.g., detecting data drift or concept drift) to trigger automated retraining if necessary.
* **Final Evaluation on Test Dataset:** Performing inference on the actual Kaggle test dataset and generating a submission file would provide a comprehensive evaluation against the competition leaderboard.
