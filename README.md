# 🧠 Saliency-Guided CNN for Brain Tumor Classification from Transcriptomic Images

## 🧠 Model Architecture and Training

### 🔧 Setup
- Framework: PyTorch
- Architectures:
  - [x] ResNet-18 (pretrained)
  - [x] EfficientNet-B0 (pretrained)
- Input Image Size: `128x128`
- Optimizer: `Adam`
- Loss Function: `CrossEntropyLoss`
- Batch Size: `32`
- Epochs: `5` (adjustable)
- Cross-Validation: `5-fold Stratified`

### 📁 Dataset Format
Structured as a directory of class folders:
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

- **GSM3828672 Smartseq2 GBM**  
  GBM subtype expression profiles labeled via `tumour name`.

---

## ⚙️ Installation

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

Clone the repo and install dependencies:

```bash
git clone https://github.com/alyatimi/Image-Based-Deep-Learning-for-Brain-Tumour-Transcriptomics.git
cd tumor-classification-cnn
pip install -r requirements.txt
```

## CITATION 
