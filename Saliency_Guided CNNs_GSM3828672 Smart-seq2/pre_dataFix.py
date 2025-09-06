import scanpy as sc
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def load_data(filepath):
    """Load raw expression data as AnnData."""
    try:
        adata = sc.read_h5ad(filepath)
        print("[INFO] Loaded data with shape:", adata.shape)
        return adata
    except Exception as e:
        print("[ERROR] Failed to load data:", e)
        raise


def compute_manifold_embeddings(adata):
    """Compute PCA, UMAP, and t-SNE and store in adata.obsm."""
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
    sc.tl.umap(adata, n_components=2)
    sc.tl.tsne(adata, n_pcs=50)  
    return adata


def map_embeddings_to_rgb(adata, image_size=128):
    """Normalize embeddings and map to RGB color and pixel grid."""
    tsne = adata.obsm["X_tsne"]
    pca = adata.obsm["X_pca"][:, :2]
    umap = adata.obsm["X_umap"]

    def normalize_coords(coords):
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        return (coords - min_vals) / (max_vals - min_vals + 1e-8)

    tsne_norm = normalize_coords(tsne)
    pca_norm = normalize_coords(pca)
    umap_norm = normalize_coords(umap)

    rgb_coords = np.stack([tsne_norm, pca_norm, umap_norm], axis=-1) 
    pixel_coords = (tsne_norm * (image_size - 1)).astype(int)
    return pixel_coords, rgb_coords


def generate_rgb_images(pixel_coords, rgb_coords, image_size=128, point_size=3):
    """Render RGB images with a colored blob at each pixel position."""
    num_samples = rgb_coords.shape[0]
    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.float32)

    for i in range(num_samples):
        x, y = pixel_coords[i]
        for dx in range(-point_size, point_size + 1):
            for dy in range(-point_size, point_size + 1):
                xi = np.clip(x + dx, 0, image_size - 1)
                yi = np.clip(y + dy, 0, image_size - 1)
                images[i, yi, xi] = rgb_coords[i, 0]  # RGB triplet
    return images


def save_rgb_images_by_label(images, labels, save_dir="benchmarking"):
    """Save each RGB image to a folder named after its tumor label."""
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        img_uint8 = (img * 255).astype(np.uint8)
        image = Image.fromarray(img_uint8)
        label_dir = os.path.join(save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image.save(os.path.join(label_dir, f"sample_{i}.png"))


def plot_pixel_density_by_label(pixel_coords, labels, image_size=128):
    """Plot density heatmap per tumor type (subplot per label)."""
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, label in enumerate(unique_labels):
        density = np.zeros((image_size, image_size), dtype=int)
        label_coords = pixel_coords[labels == label]

        for x, y in label_coords:
            x = np.clip(x, 0, image_size - 1)
            y = np.clip(y, 0, image_size - 1)
            density[y, x] += 1

        sns.heatmap(density, ax=axes[idx], cmap="viridis", cbar=False)
        axes[idx].set_title(f"{label} (n={len(label_coords)})", fontsize=10)
        axes[idx].axis("off")

    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout(pad=2.0)
    plt.savefig("Figure_1.png")
    plt.show()
def plot_combined_pixel_density(pixel_coords, labels, image_size=128):
    """Plot a combined tumor type map with a light grid background."""
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    density_map = np.zeros((image_size, image_size), dtype=int)

    for (x, y), label in zip(pixel_coords, labels):
        x = np.clip(x, 0, image_size - 1)
        y = np.clip(y, 0, image_size - 1)
        density_map[y, x] = label_to_index[label] + 1  # 0 = background

    # Grid background
    grid_bg = np.ones((image_size, image_size)) * 0.9  # light gray

    # Create color map
    cmap = ListedColormap(["white"] + list(plt.cm.tab20.colors[:len(unique_labels)]))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Show grid background
    ax.imshow(grid_bg, cmap="gray", interpolation="none")

    # Overlay tumor map
    tumor_mask = density_map > 0
    tumor_display = np.ma.masked_where(~tumor_mask, density_map)
    im = ax.imshow(tumor_display, cmap=cmap, interpolation="none")

    # Draw grid lines
    ax.set_xticks(np.arange(-0.5, image_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, image_size, 1), minor=True)
    ax.grid(which="minor", color='lightgray', linestyle='-', linewidth=0.25)
    ax.tick_params(which="minor", size=0)

    # Turn off axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Colorbar with tumor names
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(unique_labels) + 1))
    cbar.ax.set_yticklabels(["Background"] + list(unique_labels))
    cbar.set_label("Tumor Type")

    ax.set_title("Combined Tumor Type Pixel Map")
    plt.tight_layout()
    plt.savefig("Figure_combined_tumor.png")
    plt.show()



def plot_pixel_density(pixel_coords, image_size=128, title="Genes per pixel"):
    """Plot heatmap of how many samples fall into each pixel."""
    density = np.zeros((image_size, image_size), dtype=int)
    for x, y in pixel_coords:
        x = np.clip(x, 0, image_size - 1)
        y = np.clip(y, 0, image_size - 1)
        density[y, x] += 1

    plt.figure(figsize=(8, 8))
    sns.heatmap(density, cmap="viridis", cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("Figure_genes_pixel.png")
    plt.show()


def main():
    adata = load_data("data/Smartseq2_upperlim_gbm_data.h5ad")
    adata = compute_manifold_embeddings(adata)

    pixel_coords, rgb_coords = map_embeddings_to_rgb(adata, image_size=128)
    images = generate_rgb_images(pixel_coords, rgb_coords, image_size=128)
    labels = adata.obs["tumour name"].values

    save_rgb_images_by_label(images, labels)
    plot_pixel_density_by_label(pixel_coords, labels)
    plot_combined_pixel_density(pixel_coords, labels)
    plot_pixel_density(pixel_coords, image_size=128)


if __name__ == "__main__":
    main()
