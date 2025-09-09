# 🎭 Multi-Modal Emotion Recognition using Deep Learning

> A deep learning-based framework for **multi-modal emotion recognition** leveraging **audio** and **textual** features, achieving **70.49% accuracy** on the **MELD dataset**.  
This project combines **MobileBERT-based textual embeddings** and **MFCC-based audio features** with **custom CNN architectures** for multimodal fusion.

---

The code is in the zip folder named MMERonMELD as a .ipynb file 
## 📌 Overview

Understanding human emotions through multiple modalities has numerous real-world applications, including **mental health monitoring**, **customer sentiment analysis**, **education technology**, and **human-computer interaction**.

This project proposes a **multi-modal deep learning pipeline** that:
- Extracts **audio features** using **Mel-Frequency Cepstral Coefficients (MFCCs)**.
- Extracts **text embeddings** using a **pre-trained MobileBERT** transformer.
- Passes features through **parallel CNN-based sub-models**.
- Performs **multimodal fusion** using concatenation followed by fully connected layers.
- Classifies emotions into **seven distinct classes**.

---

## 🧠 Features

- **Multi-modal Fusion**: Combines speech and text features for enhanced performance.
- **MobileBERT Integration**: Lightweight transformer for efficient textual representation.
- **MFCC-based Audio Processing**: Captures detailed frequency-based audio features.
- **Custom CNN Architecture**: Parallel sub-models for each modality.
- **Optimized Training**:
  - Adam optimizer with a learning rate scheduler.
  - Categorical cross-entropy loss.
  - Batch normalization to prevent overfitting.
- **High Accuracy**: Achieved **70.49% validation accuracy**, outperforming prior approaches.

---

## 📊 Dataset

### **MELD Dataset**  
[MELD: Multimodal EmotionLines Dataset](https://github.com/SenticNet/MELD)  

- Derived from the TV series **Friends**.
- Contains **1,400+ dialogues** and **13,000+ utterances**.
- Includes **audio, text, and visual modalities**.
- Emotion categories:
  - `Anger`, `Disgust`, `Fear`, `Joy`, `Neutral`, `Sadness`, `Surprise`
- Sentiment labels: **positive**, **negative**, **neutral**.

---

## 🏗️ Methodology

### **1. Audio Feature Extraction**
- Converted `.mp4` files to `.wav` using **PyDub**.
- Extracted **MFCCs** using **Librosa**.
- Generated fixed-length vectors per utterance via **zero-padding**.

### **2. Text Feature Extraction**
- Used **MobileBERT** for generating contextual embeddings.
- Tokenized sequences, padded them, and extracted **last hidden state representations**.

### **3. Model Architecture**

**Pipeline Summary**:
- **Audio Sub-network**: CNN with Dense + BatchNorm layers.
- **Text Sub-network**: CNN with Dense + BatchNorm layers.
- **Fusion Layer**: Concatenation → Dense layers → Softmax classification.

📌 **Suggested Figure** → Add a **model architecture diagram** here  
(_Recommended_: Use Fig.14 from your thesis for clarity.)

---

## ⚙️ Implementation Details

| Component       | Configuration                        |
|-----------------|-------------------------------------|
| **Frameworks**  | TensorFlow, PyTorch, Librosa, PyDub |
| **Text Model**  | MobileBERT (pre-trained)            |
| **Audio Model** | MFCC-based CNN                      |
| **Fusion**      | Concatenation + Dense Layers        |
| **Optimizer**   | Adam + LR Scheduler                 |
| **Loss**        | Categorical Cross-Entropy           |
| **Dataset Split** | 80% train / 20% test               |
| **Accuracy**    | **70.49%**                          |

---

## 📈 Results

| Method                                | Accuracy (%) |
|--------------------------------------|--------------|
| DialogueCRN (Hu et al., 2021)        | 60.73        |
| UniMSE (Hu et al., 2022)             | 65.00        |
| M2FNet (Chudasama et al., 2022)      | 67.85        |
| **Our Proposed Method**             | **70.49**    |

📌 **Suggested Figure** → Include training & validation **loss curves** here  
(_Use Fig.15 and Fig.16 from your thesis_).

---

## 🚀 Installation & Usage

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/multi-modal-emotion-recognition.git
cd multi-modal-emotion-recognition




