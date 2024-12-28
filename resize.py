import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist
    
    # Get a list of all image files (supports jpg, jpeg, png)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        # Construct the full file path
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)
        
        # Open the image
        with Image.open(input_image_path) as img:
            # Convert to 'RGB' if the image is in 'P' mode or 'RGBA' mode
            if img.mode in ['P', 'RGBA']:
                img = img.convert('RGB')
            
            # Resize the image to the target size
            img_resized = img.resize(target_size)
            
            # Save the resized image to the output folder
            img_resized.save(output_image_path)
            print(f"Resized and saved: {image_file}")

# Example usage:
input_folder = './crushit'  # Folder containing images
output_folder = './crushit2'  # Folder to save resized images

resize_images_in_folder(input_folder, output_folder, target_size=(256, 256))
