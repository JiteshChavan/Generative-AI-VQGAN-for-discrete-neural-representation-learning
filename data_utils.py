import os
import numpy as np
from PIL import Image
from torchvision import transforms
import glob

import torch



from dataclasses import dataclass

@dataclass
class Data_Utils_Config:
    downscale_res : int = 256

    # path to downscaled and cropped images
    src_processed : str = "./test"
    # dest path for shards
    dest_shards : str = "./testShards"

    # images to be packed in a single shard
    shard_batch_size : int = 640

    # reconstruction types
    recon_types = {"clone", "neural_recon"}



class DataUtils:
    def __init__(self, util_config):
        self.util_config = util_config

    # ------------------------------------------------------------------------------------------------------------------
    # Process iamges in src_path and save them as npy shards in out_path, each shard containing batch_size images
    # -------------------------------------------------------------------------------------------------------------------
    def normalize_image(self, image):
        """
        Normalize image pixel values to be between -1 and 1.
        
        Parameters:
            image (PIL.Image.Image): Image object to normalize.
            
        Returns:
            np.ndarray: Normalized image array.
        """
        image_np = np.array(image).astype(np.float16)  # Convert to numpy array
        return (image_np / 127.5) - 1.0  # Normalize to [-1, 1]

    def normalized_images_to_shards(self, src_pre_procced, dest_shards, shard_batch_size):
        """
        Process images in the folder, normalize them, and save them into numpy shards.
        
        Parameters:
            src_path (str): Folder containing the images.
            batch_size (int): Number of images per shard.
            out_path (str): Folder to save the .npy files.
        """
        # Ensure the output directory exists
        os.makedirs(dest_shards, exist_ok=True)
        
        # Get all image files in the folder (supports .jpg, .png)
        image_paths = glob.glob(os.path.join(src_pre_procced, "*.jpg")) + glob.glob(os.path.join(src_pre_procced, "*.png"))
        
        images = []  # List to store images for each batch
        shard_index = 0  # Index for naming the .npy files

        # Iterate through all images
        for image_path in image_paths:
            # Open image and convert to RGB
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Normalize the image
            normalized_image = self.normalize_image(image)
            
            # Add the image to the list
            images.append(normalized_image)
            
            # If we reach batch size, save the batch and reset the list
            if len(images) == shard_batch_size:
                batch_array = np.stack(images)  # Stack images into a single numpy array (B, H, W, C)
                np.save(os.path.join(dest_shards, f"shard_{shard_index:03d}.npy"), batch_array)
                print(f"Saved shard {shard_index:03d} with {shard_batch_size} images.")
                
                # Reset for next batch
                images = []
                shard_index += 1
        
        # Save any remaining images (if batch size is not a perfect multiple)
        if images:
            batch_array = np.stack(images)
            np.save(os.path.join(dest_shards, f"images_batch_{shard_index:03d}.npy"), batch_array)
            print(f"Saved shard {shard_index:03d} with remaining images.")
        
        print(f"All images processed and saved in {dest_shards}.")


    # ------------------------------------------------------------------------------------------------------------------------------
    # Reconstruct image from a tensor (B, C, H, W) normalized to [-1,1] and store it to out_path, with specified recon_type tag
    # ------------------------------------------------------------------------------------------------------------------------------
    def tensor_to_image (self, tensor, out_path, recon_type):
        """
        Reconstruct and save images from a PyTorch tensor.

        Parameters:
            tensor (torch.Tensor): Tensor of shape (B, C, H, W) with pixel values normalized to [-1, 1].
            output_folder (str): Folder to save the reconstructed images.
        """
        # ensure the tensor values are in range [-1, 1]
        assert torch.min(tensor) >= -1 and torch.max(tensor) <= 1, f"Tensor values must be in [-1,1] range"
        assert recon_type in self.util_config.recon_types

        if recon_type == "clone":
            recon_tag = "clone"
        else:
            recon_tag = "neural_recon"

        os.makedirs(out_path, exist_ok=True)

        # Denormalize the tensor to [0, 255]
        tensor = ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8) # convert to uint8

        # permute tensor back to (B H W C) from B C H W
        tensor = tensor.permute (0, 2, 3, 1).cpu().numpy()
        for i, image in enumerate (tensor):
            image = Image.fromarray (image) # convert to PIL Image
            image.save (f"{out_path}/{recon_tag}_{i:04d}.png")
            print (f"saved {out_path}/{recon_tag}_{i:04d}.png")


    # ------------------------------------------------------------------------------------------------------------------------------
    # Scale down the smaller res of images to 256
    # ------------------------------------------------------------------------------------------------------------------------------


    def preprocess_image(self, image_path):
        """
        Preprocess an image to maintain the original aspect ratio and apply a 256x256 center crop.
        
        Parameters:
            image_path (str): Path to the input image.
            
        Returns:
            PIL.Image.Image: Preprocessed image.
        """
        # Define the transformation to resize the smaller dimension and center crop to 256x256
        transform = transforms.Compose([
            transforms.Resize(self.util_config.downscale_res),  # Resize the smaller dimension to 256 (maintains aspect ratio)
            transforms.CenterCrop(self.util_config.downscale_res)  # Center crop to 256x256
        ])
        
        # Load the image
        image = Image.open(image_path)
        
        # Convert to RGB if the image is not already in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply the transformations
        processed_image = transform(image)
        
        return processed_image

    def preprocess_images_in_folder(self, src, dest):
        """
        Process all images in a specified folder and save them after applying the preprocessing.
        
        Parameters:
            input_folder (str): Folder containing the input images.
            output_folder (str): Folder where the processed images will be saved.
        """
        # Create output folder if it doesn't exist
        os.makedirs(src, exist_ok=True)
        os.makedirs(dest, exist_ok=True)
        
        # Iterate through the images in the input folder
        for i, file_name in enumerate(os.listdir(src)):
            # Check for image files (can add more file types if needed)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(src, file_name)
                
                # Preprocess the image
                processed_image = self.preprocess_image(image_path)
                
                # Save the processed image to the output folder
                output_path = os.path.join(dest, file_name)
                processed_image.save(output_path)
        print(f"{i+1} Images Processed and saved to: {dest} (assuming all were images in the 3 supported format)")

