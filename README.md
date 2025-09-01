# 🧠 Saliency-Guided CNN for Brain Tumor Classification from Transcriptomic Images

This repository implements a pipeline that transforms single-cell RNA sequencing (scRNA-seq) data into RGB images using dimensionality reduction and spectral mapping. A ResNet-18 CNN is trained to classify brain tumor subtypes, and saliency maps are used for model interpretability.

---

## 📂 Project Structure

.
├── data/ # Raw .h5ad transcriptomic files
│ ├── GSE85217_data.h5ad # Dataset with 28 tumor subtypes
│ └── Smartseq2_upperlim_gbm_data.h5ad # GBM-specific dataset
├── benchmarking/ # Generated RGB images, organized by class
├── pre_dataFix.py # Preprocessing: embeddings → RGB images
├── cnn_model.py # CNN training and evaluation
├── generate_saliency.py # Saliency map computation
├── training_history.png # Loss plots
├── Figure_*.png # Confusion matrices, saliency, heatmaps
└── README.md # Project documentation


---

## 🧬 Datasets

- **GSE85217**  
  28 annotated brain tumor subtypes; preprocessed from `.h5ad`.

- **Smartseq2 GBM**  
  GBM subtype expression profiles labeled via `tumour name`.

---

## ⚙️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/tumor-classification-cnn.git
cd tumor-classification-cnn
pip install -r requirements.txt
