import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def plot_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    if not image_files:
        print("No images found in the folder.")
        return
    
    images = [Image.open(os.path.join(folder_path, img)) for img in image_files[:12]]  # Limit to 12 images
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))  # 2 rows, 6 columns
    axes = axes.flatten()
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot images in a folder in two rows, each containing 6 images.")
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    args = parser.parse_args()
    
    plot_images(args.folder)
