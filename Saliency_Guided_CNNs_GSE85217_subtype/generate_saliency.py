import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128
MODEL_PATH = "model_final.pth"
IMAGE_DIR = "benchmarking"

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)  # MATCH number of classes in saved model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def compute_saliency(model, input_tensor):
    input_tensor.requires_grad_()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    output[0, pred_class].backward()
    saliency = input_tensor.grad.abs().squeeze().max(dim=0)[0].cpu().numpy()
    return saliency

# Gather class names
class_names = sorted([f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))])

# Collect saliency maps
saliency_maps = []
valid_class_names = []

for class_name in class_names:
    class_dir = os.path.join(IMAGE_DIR, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith(".png")]
    if not image_files:
        continue

    image_path = os.path.join(class_dir, image_files[0])
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        saliency = compute_saliency(model, input_tensor)
        saliency_maps.append(saliency)
        valid_class_names.append(class_name)
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")

# Plot batches of 4 saliency maps
def plot_in_batches(saliency_maps, class_names, batch_size=4):
    total = len(saliency_maps)
    for i in range(0, total, batch_size):
        fig, axes = plt.subplots(1, batch_size, figsize=(4 * batch_size, 4))
        for j in range(batch_size):
            idx = i + j
            if idx < total:
                ax = axes[j]
                ax.imshow(saliency_maps[idx], cmap="hot")
                ax.set_title(class_names[idx], fontsize=12, fontweight='bold')
                ax.axis("off")
            else:
                axes[j].axis("off")
        plt.tight_layout()
        plt.savefig(f"Figure_Saliency_{i//batch_size + 1}.png", dpi=300)
        plt.show()

plot_in_batches(saliency_maps, valid_class_names)
