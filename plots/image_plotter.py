import os
import matplotlib.pyplot as plt
from PIL import Image

def plot_images_two_rows(folder_path, max_images=None):
    """
    Plots images from a specified folder in two rows:
    - The first row contains "actual<number>" images.
    - The second row contains "neural<number>" images.
    
    Args:
        folder_path (str): Path to the folder containing the images.
        max_images (int, optional): Maximum number of pairs to display. Defaults to None (display all).
    """
    # Collect and sort image files
    image_files = sorted(os.listdir(folder_path))
    
    # Filter "actual" and "neural" images
    actual_images = sorted([f for f in image_files if f.startswith("actual")])
    neural_images = sorted([f for f in image_files if f.startswith("neural")])
    
    # Match the number of images in both rows
    num_images = min(len(actual_images), len(neural_images))
    
    if num_images == 0:
        print("No valid 'actual' and 'neural' images found in the folder.")
        return

    # Apply the max_images limit if specified
    if max_images:
        num_images = min(num_images, max_images)
        actual_images = actual_images[:num_images]
        neural_images = neural_images[:num_images]

    # Plotting
    fig, axs = plt.subplots(2, num_images, figsize=(5 * num_images, 10))

    for i in range(num_images):
        # Load the actual and neural images
        actual_path = os.path.join(folder_path, actual_images[i])
        neural_path = os.path.join(folder_path, neural_images[i])
        
        actual_image = Image.open(actual_path)
        neural_image = Image.open(neural_path)
        
        # Plot "actual" image in the first row
        axs[0, i].imshow(actual_image)
        axs[0, i].axis("off")  # Turn off axis

        # Plot "neural" image in the second row
        axs[1, i].imshow(neural_image)
        axs[1, i].axis("off")  # Turn off axis

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    folder_path = "plotpairsepoch0"  # Replace with your folder path
    plot_images_two_rows(folder_path)  # No max_images by default
