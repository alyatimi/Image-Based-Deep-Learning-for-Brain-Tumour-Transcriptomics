import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torchvision.models import efficientnet_b0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "benchmarking"
BATCH_SIZE = 32
IMAGE_SIZE = 128
NUM_EPOCHS = 5
K_FOLDS = 5


def get_dataset(data_dir, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset




def build_model(model_type, num_classes):
    if model_type == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model type")
    return model.to(device)



def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate_model_full(model, dataloader, class_names):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Compute OvO ROC-AUC scores
    auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='weighted')

    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return acc, precision, recall, f1, auc_macro, auc_weighted, cm


def plot_training_history(train_losses):
    plt.figure(figsize=(8, 5))
    for i, loss_curve in enumerate(train_losses):
        plt.plot(loss_curve, label=f"Fold {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Fold GSE85217")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


def plot_confusion_matrix(cm, class_names, fold=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    title = f"Confusion Matrix GSE85217"
    if fold is not None:
        title += f" (Fold {fold + 1})"
    plt.title(title)
    plt.tight_layout()
    if fold is not None:
        plt.savefig(f"confusion_matrix_fold_{fold+1}.png")
    else:
        plt.savefig("confusion_matrix_avg.png")
    plt.show()

def main():
    dataset = get_dataset(DATA_DIR, IMAGE_SIZE)
    class_names = dataset.classes
    targets = [label for _, label in dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    results = {}  # Store metrics for both models

    for model_type in ["resnet", "efficientnet"]:
        print(f"\n=== Evaluating {model_type.upper()} ===")

        all_metrics = []
        all_loss_curves = []
        all_conf_matrices = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
            print(f"[{model_type.upper()}] Starting Fold {fold + 1}/{K_FOLDS}")
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            model = build_model(model_type, num_classes=len(class_names))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            fold_loss_curve = []
            for epoch in range(NUM_EPOCHS):
                epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)
                fold_loss_curve.append(epoch_loss)
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

            all_loss_curves.append(fold_loss_curve)

            try:
                acc, precision, recall, f1, auc_macro, auc_weighted, cm = evaluate_model_full(model, val_loader,
                                                                                              class_names)
            except Exception as e:
                print(f"[{model_type.upper()}] Error in evaluation: {e}")
                continue

            all_metrics.append((acc, precision, recall, f1, auc_macro, auc_weighted))
            all_conf_matrices.append(cm)

        # Store metrics
        results[model_type] = {
            "metrics": all_metrics,
            "loss_curves": all_loss_curves,
            "conf_matrices": all_conf_matrices,
        }

        # Print average metrics for current model
        print(f"\nAverage Metrics for {model_type.upper()}:")
        metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC Macro", "ROC-AUC Weighted"]
        mean_vals = np.mean(all_metrics, axis=0)
        for name, val in zip(metric_names, mean_vals):
            print(f"{name}: {val:.4f}")

        # Plot training loss
        plot_training_history(all_loss_curves)

        # Plot average confusion matrix
        avg_cm = np.mean(all_conf_matrices, axis=0).round().astype(int)
        plot_confusion_matrix(avg_cm, class_names)

    # Wilcoxon Signed-Rank Test (Accuracy Comparison)
    from scipy.stats import wilcoxon
    acc_resnet = [m[0] for m in results["resnet"]["metrics"]]
    acc_efficientnet = [m[0] for m in results["efficientnet"]["metrics"]]
    stat, p = wilcoxon(acc_resnet, acc_efficientnet)
    print(f"\nWilcoxon Test (ResNet vs EfficientNet Accuracy): statistic={stat:.4f}, p-value={p:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'model_final.pth')
    print("\nModel saved to model_final.pth")


if __name__ == "__main__":
    main()
