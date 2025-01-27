import os
import numpy as np
from PIL import Image
from torchvision import transforms
import glob
import random

import torch



from dataclasses import dataclass

# TODO: Permute while forming shards so that val shard has balanced representation


@dataclass
class Data_Utils_Config:
    downscale_res : int = 256

    # reconstruction types
    recon_types = {"clone", "neural"}


class DataUtils:
    def __init__(self, util_config):
        self.util_config = util_config
        # to maintain state of how many images were reconstructed
        self.current_clone_recon_image = 0
        self.current_neural_recon_image = 0

    def reset (self):
        self.current_clone_recon_image = 0
        self.current_neural_recon_image = 0

    # ------------------------------------------------------------------------------------------------------------------------------
    # Reconstruct image from a tensor (B, C, H, W) normalized to [-1,1] and store it to dest_path, with specified recon_type tag
    # ------------------------------------------------------------------------------------------------------------------------------
    def tensor_to_image (self, tensor, dest_path, recon_tag):
        """
        Reconstruct and save images from a PyTorch tensor.

        Parameters:
            tensor (torch.Tensor): Tensor of shape (B, C, H, W) with pixel values normalized to [-1, 1].
            output_folder (str): Folder to save the reconstructed images.
        """
        # ensure the tensor values are in range [-1, 1]
        assert torch.min(tensor) >= -1 and torch.max(tensor) <= 1, f"Tensor values must be in [-1,1] range"
        assert recon_tag in self.util_config.recon_types

        if recon_tag == "clone":
            image_index = self.current_clone_recon_image
        else:
            image_index = self.current_neural_recon_image

        os.makedirs(dest_path, exist_ok=True)

        # Denormalize the tensor to [0, 255]
        tensor = ((tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8) # convert to uint8

        # permute tensor back to (B H W C) from B C H W
        tensor = tensor.permute (0, 2, 3, 1).cpu().numpy()
        for i, image in enumerate (tensor):
            image = Image.fromarray (image) # convert to PIL Image
            image.save (f"{dest_path}/{recon_tag}_{image_index:04d}.png")
            #print (f"saved {dest_path}/{recon_tag}_{image_index:04d}.png")
            image_index += 1

        if recon_tag == 'clone':
            self.current_clone_recon_image = image_index
        else:
            self.current_neural_recon_image = image_index



    # ------------------------------------------------------------------------------------------------------------------------------
    # TODO: Resizes the smallest dimension of the images in specified src to 256, maintainingthe aspect ratio, then takes center crop 256x256, converts to tensor
    # normalizes to -1 and 1 and stores in shards to specified dest, each shard containing shard_batch_size images
    # ------------------------------------------------------------------------------------------------------------------------------

    
    def process_images_in_folder (self, src, dest, shard_batch_size):
        assert os.path.exists (src)
        transform = transforms.Compose ([
            transforms.Resize (256), # downscale while maintaining aspect ratio
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        os.makedirs (dest, exist_ok=True)

        # list all the images in the source
        image_files = [os.path.join(src, i) for i in os.listdir(src) if i.lower().endswith(('png', 'jpg', 'jpeg'))]
        random.shuffle (image_files)
        shard = []
        shard_index = 0
    
        for i, image_file in enumerate (image_files):
            
            if shard_index == 0:
                shard_tag = 'val'
            else:
                shard_tag = 'train'
            
            try:
                # open and transform the image
                image = Image.open (image_file).convert('RGB') # Ensure all iamges are 3-Channel RGB
                tensor = transform(image)
                shard.append (tensor)

                # save shard if it contains shard_batch_size images
                if len(shard) == shard_batch_size:
                    shard_tensor = torch.stack(shard) # stack images in the shard
                    shard_path = os.path.join (dest, f"shard_{shard_tag}_{shard_index:04d}.npy")
                    np.save (shard_path, shard_tensor.numpy())
                    print (f"saved: shard_{shard_tag}_{shard_index:04d} with {len(shard)} images")
                    shard_index += 1
                    shard = [] # reset buffer for next shard
            except Exception as e:
                print (f"Error processing {image_file} : {e}")
        
        # save remaining images to final shard
        if len(shard) > 0:
            shard_tensor = torch.stack (shard)

            shard_path = os.path.join (dest, f"shard_{shard_tag}_{shard_index}.npy")
            np.save (shard_path, shard_tensor.numpy())
            print(f"Final shard saved with {len(shard)} images.")
        print (f"Shards saved in {dest}")
