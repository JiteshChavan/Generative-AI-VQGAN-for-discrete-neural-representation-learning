import os
from torchvision import transforms
from PIL import Image

def process_images(input_folder, output_folder, output_size=256):
    """
    Resizes all images in the input folder so that the smaller side is `output_size` pixels,
    while maintaining aspect ratio, and applies a center crop of `output_size x output_size`.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save processed images.
        output_size (int): Target size for the smaller side and the center crop. Default is 256.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the torchvision transformation
    transform = transforms.Compose([
        transforms.Resize(output_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(output_size)
    ])

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            print(f"Skipping non-image file: {filename}")
            continue

        try:
            # Open and transform the image
            image = Image.open(input_path).convert("RGB")
            processed_image = transform(image)

            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, filename)
            processed_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "actuals"  # Path to the folder with input images
    output_folder = "./"  # Path to save processed images

    process_images(input_folder, output_folder)
