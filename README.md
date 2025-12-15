# Image Captioning for Remote Sensing (RSICD Dataset)

This project implements **end-to-end image captioning for remote sensing imagery** using both **CNN + LSTM** and **CNN + Transformer** architectures. The goal is to generate descriptive captions for aerial and satellite images, capturing land use, structures, and scene layout.  

All results, analyses, and insights are available in the provided Jupyter Notebook (`Image_Captioning.ipynb`).

---

## 📂 Project Overview

**Course:** EE 782 – Advanced Topics in Machine Learning  
**Institution:** IIT Bombay  

This assignment focuses on:

1. Implementing two captioning models:  
   - CNN Encoder + LSTM Decoder  
   - CNN Encoder + Transformer Decoder  

2. Designing experiments and ablations such as:  
   - Backbone comparisons (ResNet-18 vs MobileNet)  
   - Vision-text interface variations (init hidden vs `<img>` token, memory token length)  
   - Input augmentations and decoder regularization  

3. Comprehensive **evaluation and analysis**, including:  
   - BLEU-4 and METEOR metrics  
   - Caption length statistics and degenerate repetition analysis  
   - Qualitative success/failure examples and error slices  
   - Explainability via Grad-CAM and token/attention analyses  

---

## 📦 Dataset

- **Dataset:** RSICD (Remote Sensing Image Captioning Dataset) from Kaggle  
- **Size:** ~10.9k RGB images  
- **Captions:** 5 captions per image  
- **Splits:** Train: 8,000 | Validation: 1,500 | Test: 1,421  

**Preprocessing:**

- Resize images to 224×224  
- ImageNet normalization  
- Word-level tokenizer with ~10k vocab (built on training set only)  
- Special tokens: `<bos>`, `<eos>`, `<pad>`  
- Maximum caption length: 22–24 tokens  

---

## 🖥 Model Architectures

### CNN Encoder
- ResNet-18 and MobileNet backbones (ImageNet pretrained)  
- Global average pooling  
- Feature-cache and end-to-end fine-tuning modes  

### LSTM Decoder
- Embedding dimension: 300–512  
- 1–2 LSTM layers with hidden size 512  
- Image features injected as initial hidden state or `<img>` token  
- Teacher-forcing training with cross-entropy loss  

### Transformer Decoder
- `nn.TransformerDecoder` with 2–4 layers, 4–8 attention heads, d_model=512  
- Image features projected to 1–4 memory tokens  
- Causal and key padding masks implemented  

---

## 📊 Experiments & Analysis

- Backbone swap experiments (ResNet-18 vs MobileNet)  
- Input augmentations (rotation invariance)  
- LSTM: init hidden vs `<img>` token comparison  
- Transformer: memory token length study  
- Evaluation via BLEU-4, METEOR, caption length stats, and degenerate repetition analysis  
- Grad-CAM visualizations on selected success and failure cases  

All plots, tables, and insights are in the notebook.  

---

## 📓 Notebook

- Contains full code, explanations, observations, and results  
- Includes Abstract, Introduction, Methods, Results, Discussion, and Conclusions  

---

## ⚡ Usage

1. Clone the repository:  
   ```bash
   git clone https://github.com/MedhanshSharma2004/image_captioning_rsicd.git
   cd image_captioning_rsicd
