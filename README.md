![Özyeğin University Logo](vertopal_4532c36a44ec40f3b294b7a1efbe668d/media/image1.jpeg)

---

# Özyeğin University

**Faculty of Industrial Engineering**
**Department of Data Science**

---

### **CS 566 – Senior Project Report**

**Fall 2024–2025**

---

# 🥔 Potato Leaf Disease Classification Using Deep Learning Models

**By:**
**Demba Sow**
[dastech1998@gmail.com](mailto:dastech1998@gmail.com)

**Supervised by:**
**Prof. Dr. Hasan Fehmi Ateş**

---

## 📄 Abstract

Potato crops are vital for global food security, making disease detection essential. This project presents a deep learning-based classification of potato leaf diseases using CNN, MobileNetV2, ResNet50, and DenseNet121. Experiments were conducted on the Potato Leaf Disease Dataset (PLD), and models were evaluated using accuracy, loss, precision, recall, and F1-score.

---

## 🌱 Introduction

Potatoes are a major food source globally. Diseases such as early blight and late blight threaten crop yields and farmers' livelihoods. Traditional detection methods are laborious and error-prone, necessitating intelligent, automated solutions.

### 🎯 Objectives

* Apply deep learning models to classify potato leaf diseases.
* Compare performance and apply fine-tuning.
* Recommend the most effective model based on performance metrics.

---

## 🔬 Methodology

### 📁 Dataset Description

The **Potato Leaf Disease Dataset (PLD)** contains 4,072 images from Central Punjab, Pakistan, categorized into:

* **Early Blight**: Brown/black lesions from fungal infection.
* **Late Blight**: Water-soaked spots caused by microorganisms.
* **Healthy**: Leaves with no signs of disease.

#### 🖼 Sample Images

*Early Blight*
![Early Blight](media/image2.jpeg)

*Late Blight*
![Late Blight](media/image3.jpeg)

*Healthy*
![Healthy](media/image4.jpeg)

#### Dataset Structure

| Subset     | Early Blight | Late Blight | Healthy  | Total    |
| ---------- | ------------ | ----------- | -------- | -------- |
| Training   | 1303         | 1132        | 816      | 3251     |
| Validation | 163          | 151         | 102      | 416      |
| Testing    | 162          | 141         | 102      | 405      |
| **Total**  | **1628**     | **1424**    | **1020** | **4072** |

---

### 🧼 Data Preprocessing

* ✅ Corrupted image check: None found
* 📏 Resized all images to **224x224**
* ⚙️ Normalized pixel values to \[0, 1]
* 🔁 Applied data augmentation: rotation, zoom, flipping, brightness adjustments

---

## 🧠 Deep Learning Models

### 📌 Custom CNN I – 8 Layers

* 2 Conv2D + MaxPooling layers
* 1 Flatten layer
* 1 Dropout (50%)
* 1 Dense layer (128 neurons)
* Output layer (3 neurons)

![CNN I](media/image5.jpeg)
*3D Architecture View*
![CNN I 3D](media/image6.png)

---

### 📌 Custom CNN II – 17 Layers

* 6 Conv2D + MaxPooling layers
* 1 Dropout
* 1 Flatten
* 2 Dense layers (256 & 128 neurons)
* Output layer (3 neurons)

![CNN II](media/image7.png)
*3D Architecture View*
![CNN II 3D](media/image8.png)

---

### 📌 MobileNetV2

* Lightweight and efficient
* Uses depthwise separable convolutions
* Ideal for mobile and low-latency applications

Architecture:

* MobileNetV2 base (pretrained)
* GlobalAveragePooling2D
* Dense (128 neurons)
* Dropout (50%)
* Output layer (3 neurons)

![MobileNetV2](media/image9.jpeg)
*3D Architecture View*
![MobileNetV2 3D](media/image10.png)

---

### 📌 ResNet50

* 50-layer deep residual network
* Uses skip connections to avoid vanishing gradients
* May underperform with small datasets

Architecture:

* ResNet50 base
* GlobalAveragePooling2D
* Dense (128 neurons)
* Dropout (50%)
* Output layer (3 neurons)

![ResNet50](media/image11.jpeg)
*3D Architecture View*
![ResNet50 3D](media/image12.png)

---

### 📌 DenseNet121

* Dense connectivity: every layer receives input from all previous layers
* Encourages feature reuse
* High performance but more resource-intensive

Architecture:

* DenseNet121 base
* GlobalAveragePooling2D
* Dense (128 neurons)
* Dropout
* Output layer (3 neurons)

![DenseNet121](media/image13.jpeg)
*3D Architecture View*
![DenseNet121 3D](media/image14.png)

---

### 🏋️ Training Setup

* Loss Function: Categorical Cross-Entropy
* Optimizer: Adam
* Learning Rate: 0.001 (with scheduler)
* Batch Size: 32
* Early Stopping: Enabled
* Epochs: 20

---

## 📊 Results & Analysis

### 📈 Evaluation Metrics

* **Accuracy**: Overall correct predictions
* **Loss**: Model error
* **Precision**: TP / (TP + FP)
* **Recall**: TP / (TP + FN)
* **F1-score**: Harmonic mean of precision and recall

---

### 🔎 Initial Model Results

| Model       | Accuracy | Loss | Precision | Recall | F1-score |
| ----------- | -------- | ---- | --------- | ------ | -------- |
| CNN I       | 80%      | 0.65 | 0.85      | 0.86   | 0.85     |
| CNN II      | 96%      | 0.12 | 0.97      | 0.98   | 0.98     |
| MobileNetV2 | 94%      | 0.18 | 0.95      | 0.95   | 0.95     |
| ResNet50    | 70%      | 1.04 | 0.13      | 0.33   | 0.19     |
| DenseNet121 | 95%      | 0.14 | 0.96      | 0.96   | 0.96     |

![Initial Results](media/image15.png)

---

### 🎯 Fine-Tuning Results

| Model       | Accuracy | Loss | Precision | Recall | F1-score |
| ----------- | -------- | ---- | --------- | ------ | -------- |
| CNN I       | 88%      | 0.33 | 0.89      | 0.88   | 0.88     |
| CNN II      | 96%      | 0.12 | 0.97      | 0.98   | 0.98     |
| MobileNetV2 | 94%      | 0.18 | 0.95      | 0.95   | 0.95     |
| ResNet50    | 70%      | 1.04 | 0.13      | 0.33   | 0.19     |
| DenseNet121 | 95%      | 0.14 | 0.96      | 0.96   | 0.96     |

![Fine-tuned Results](media/image16.png)

---

### 📌 Key Insights

* **CNN II** is the best model overall: High accuracy (96%) and low loss (0.12).
* **CNN I** improved from 80% to 88% after tuning.
* **MobileNetV2** and **DenseNet121** offer great trade-offs between speed and accuracy.
* **ResNet50** underperformed in all scenarios.

---

### 📊 Model Summary Table

| Model       | Architecture      | Strengths                    | Weaknesses                              |
| ----------- | ----------------- | ---------------------------- | --------------------------------------- |
| MobileNetV2 | Lightweight       | Fast, efficient, robust      | Slightly less accurate than DenseNet121 |
| ResNet50    | Deep (50 layers)  | Handles deeper networks      | Struggles with small datasets           |
| DenseNet121 | Dense connections | High feature reuse, accurate | More resource-intensive                 |

---

## 🔁 Impact of Data Augmentation

To address class imbalance and test model robustness, we augmented the dataset 10x:

| Subset     | Early Blight | Late Blight | Healthy    | Total      |
| ---------- | ------------ | ----------- | ---------- | ---------- |
| Training   | 11,320       | 11,320      | 11,320     | 33,960     |
| Validation | 1,510        | 1,510       | 1,510      | 4,530      |
| Testing    | 1,410        | 1,410       | 1,410      | 4,230      |
| **Total**  | **14,240**   | **14,240**  | **14,240** | **42,720** |

*Total size: 1.5 GB*

---

## ✅ Conclusion

This study confirms that deep learning can effectively classify potato leaf diseases, with CNN II, DenseNet121, and MobileNetV2 showing excellent performance. Future work may explore:

* Real-time mobile app deployment
* Further augmentation with GANs
* Inclusion of additional disease categories

