# 📄 **Text Image Super-Resolution**  

**Final Project for Psych 186B, UCLA**  
👤 *Eric Xia & Jiayang Li*  
📅 *March 2025*  

## 🚀 Overview  

This project explores **Text Image Super-Resolution**, a deep learning-based approach to enhance the clarity and legibility of low-resolution text images. The goal is to upscale degraded text images while preserving fine-grained details critical for readability and Optical Character Recognition (OCR).  

Our model is inspired by **ResNet** and incorporates **Residual Blocks, Upsample Blocks, and Global Skip Connections** to reconstruct high-resolution text images efficiently. We evaluate our model against various downsampling methods to assess its generalization capability.

---

## 📂 Project Structure  

The repository contains the following main files and directories:  

### 🏗 **Directories**  
- **`data_utils/`** 📦 - Contains scripts for loading, preprocessing, and augmenting datasets.  
- **`model/`** 🏗 - Includes the model architecture and training utilities.  
- **`test/test_pdfs/`** 📄 - Stores test samples for evaluation, including PDFs and processed images.

### 📜 **Notebooks & Scripts**  
- **`train_model.ipynb`** 🎯 - Core training script for the **super-resolution model**.
- **`evaluation.ipynb`** 📊 - Notebook for evaluating model performance and produce visualizations. Metrics we used include **PSNR, Character Similarity, and Word Similarity**.

---

## 📌 **How to Use**  

1. Clone the repository:  
   ```bash
   git clone https://github.com/Ruichu-Xia/text-super-resolution.git
   cd text-super-resolution
2. Try our model:
   Use our **`evaluation.ipynb`** to test your own pdf pages!
