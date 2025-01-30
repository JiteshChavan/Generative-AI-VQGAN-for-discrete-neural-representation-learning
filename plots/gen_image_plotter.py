import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def plot_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    if not image_files:
        print("No images found in the folder.")
        return
    
    images = [Image.open(os.path.join(folder_path, img)) for img in image_files]
    
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 3, 3))
    if len(images) == 1:
        axes = [axes]
    
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot images in a folder in a single row.")
    parser.add_argument("folder", type=str, help="Path to the folder containing images")
    args = parser.parse_args()
    
    plot_images(args.folder)
