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
🔄 Preprocessing Pipeline (pre_dataFix.py)

Transforms gene expression data into interpretable RGB image format.

Steps:

Load .h5ad file using Scanpy

Apply PCA (50 components), then UMAP and t-SNE

Normalize and map:

t-SNE → Red

PCA → Green

UMAP → Blue

Render each cell to a 128×128 RGB image

Save by class in /benchmarking/

🧠 Model Training (cnn_model.py)

Model: ResNet-18 (ImageNet-pretrained)

Loss: CrossEntropy

Optimizer: Adam (lr = 1e-4)

Cross-validation: 5-fold stratified

Input: RGB images (128×128)

Batch size: 32

Epochs: 5

📊 Metrics

Accuracy

Precision

Recall

F1-score

Macro AUC (OvR)

🔍 Saliency Maps (generate_saliency.py)

Visualize what the model "sees" in each tumor image.

Loads trained model_final.pth

Computes gradients w.r.t. input

Generates and saves saliency maps per class

Output files: Figure_Saliency_1.png, Figure_Saliency_2.png, etc.

📈 Sample Outputs

training_history.png – Loss per fold

Figure_combined_tumor.png – Tumor-type pixel map

confusion_matrix_avg.png – Averaged performance matrix

Figure_Saliency_*.png – Class-wise saliency interpretation

🧪 Statistical Testing

Wilcoxon Signed-Rank Test used to compare ResNet-18 vs. EfficientNet

Result: p = 0.7500 → No statistically significant difference

Conclusion: ResNet-18 performs comparably with less computational overhead
