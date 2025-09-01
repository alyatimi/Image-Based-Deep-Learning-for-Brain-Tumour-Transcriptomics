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


def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    cm = confusion_matrix(all_labels, all_preds)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    return acc, precision, recall, f1, auc, cm


def plot_training_history(train_losses):
    plt.figure(figsize=(8, 5))
    for i, loss_curve in enumerate(train_losses):
        plt.plot(loss_curve, label=f"Fold {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Fold")
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
    title = f"Confusion Matrix"
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

    all_metrics = []
    all_loss_curves = []
    all_conf_matrices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        fold_loss_curve = []
        for epoch in range(NUM_EPOCHS):
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            fold_loss_curve.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

        all_loss_curves.append(fold_loss_curve)

        acc, precision, recall, f1, auc, cm = evaluate_model_full(model, val_loader, class_names)
        all_metrics.append((acc, precision, recall, f1, auc))
        all_conf_matrices.append(cm)

    # Print averaged metrics
    print("\nAverage Metrics across Folds:")
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    mean_metrics = np.mean(all_metrics, axis=0)
    for name, val in zip(metrics_names, mean_metrics):
        print(f"{name}: {val:.4f}")

    # Optional: Wilcoxon Signed-Rank Test (e.g., comparing this model vs another)
    from scipy.stats import wilcoxon
    resnet_acc = [m[0] for m in all_metrics]  # accuracy from current model
    efficientnet_acc = [0.92, 0.91, 0.91, 0.89, 0.91]  # example values for comparison

    stat, p = wilcoxon(resnet_acc, efficientnet_acc)
    print(f"\nWilcoxon Test (Accuracy vs EfficientNet): statistic={stat:.4f}, p-value={p:.4f}")

    # Plot training loss
    plot_training_history(all_loss_curves)

    # Average confusion matrix
    avg_cm = np.mean(all_conf_matrices, axis=0).round().astype(int)
    plot_confusion_matrix(avg_cm, class_names)

    # Save final model
    torch.save(model.state_dict(), 'model_final.pth')

"""def main():
    dataset = get_dataset(DATA_DIR, IMAGE_SIZE)
    class_names = dataset.classes
    targets = [label for _, label in dataset]

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    all_metrics = []
    all_loss_curves = []
    all_conf_matrices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")


        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(num_classes=len(class_names))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        fold_loss_curve = []
        for epoch in range(NUM_EPOCHS):
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            fold_loss_curve.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

        all_loss_curves.append(fold_loss_curve)

        acc, precision, recall, f1, auc, cm = evaluate_model_full(model, val_loader, class_names)
        all_metrics.append((acc, precision, recall, f1, auc))
        all_conf_matrices.append(cm)
        # Save the trained model for this fold
        #torch.save(model.state_dict(), f'model_fold{fold + 1}.pth')

    # Print averaged metrics
    print("\nAverage Metrics across Folds:")
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
    mean_metrics = np.mean(all_metrics, axis=0)
    for name, val in zip(metrics_names, mean_metrics):
        print(f"{name}: {val:.4f}")

    # Plot training loss
    plot_training_history(all_loss_curves)

    # Average confusion matrix
    avg_cm = np.mean(all_conf_matrices, axis=0).round().astype(int)
    plot_confusion_matrix(avg_cm, class_names)
    torch.save(model.state_dict(), 'model_final.pth')

"""
if __name__ == "__main__":
    main()
