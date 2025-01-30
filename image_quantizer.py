
import os

from dataclasses import dataclass

# load images from shards
# forward pass through quantizer
# store the encoding indices in shards


@dataclass
class ImageQuantizerConfig:
    # Path to pretrained vqgan checkpoint
    checkpointPath : str = "./vqgan checkpoints/model_100000.pt"
    # Path to shardified image dataset
    srcPath : str = "./shards"
    # path to where the latent representations (collection of indices of codebook vectors that represent images in latent space) of images in dataset are shardified and stored
    destPath : str = "./shardifiedLatents" 
    # forward pass batch size
    B : int = 8
    shardBatchSize : int = 10000
    # MACROS
    START_TOKEN : int = 8192

    val_shard_size : int = 100
    train_shard_size : int = 10000

import numpy as np
def load_tokens (filename):
    npt = np.load (filename)
    npt = npt.astype (np.float32)
    ptt = torch.tensor (npt, dtype=torch.float32)
    return ptt

# for_inference flag for sending indices straight into decoder,
# removes the start tokens
def load_indices (filename, for_inference=True):
    npt = np.load (filename)
    npt = npt.astype (np.int32)
    ptt = torch.tensor (npt, dtype=torch.long)
    if for_inference:
        ptt = ptt[:,1:]
    return ptt

# decently documented in VQGAN repo train_resnet_vqgan.py
class DataloaderLite:
    def __init__(self, B, num_processes, process_rank, data_root):
        self.B = B
        self.num_processes = num_processes
        self.process_rank = process_rank
        assert os.path.exists (data_root)
        
        shards = os.listdir (data_root)
        shards = [s for s in shards]
        shards = [os.path.join(data_root, s) for s in shards]
        shards = sorted (shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found"
        if master_process:
            print (f"found {len(shards)} shards")
        self.reset ()
    
    def reset (self):
        self.current_shard = 0
        self.tokens = load_tokens (self.shards[self.current_shard])
        self.current_position = self.process_rank * self.B
        self.isZerothEpoch = True

    def next_batch(self):
        B = self.B
        # Get the remaining tokens for this batch
        remaining_tokens = len(self.tokens) - self.current_position
        batch_size = min(B, remaining_tokens)  # Adjust batch size if we're at the end of the shard

        x = self.tokens[self.current_position : self.current_position + batch_size]  # Adjusted batch size

        self.current_position += self.B * self.num_processes

        # If loading the next batch would be out of bounds, reset
        if self.current_position >= len(self.tokens):  # Reached the end of the current shard
            if self.current_shard + 1 == len(self.shards):  # If it's the last shard
                self.isZerothEpoch = False
                # Handle the remaining tokens for the final shard
                if len(x) < B and len(self.tokens[self.current_position:]) < B:
                    return x  # Return whatever remains

            self.current_shard = (self.current_shard + 1) % len(self.shards)
            print(f"Moving onto shard: {self.current_shard}")
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.process_rank * B

        return x





import torch
import torch.nn as nn
import torch.nn.functional as F
class ImageQuantizer (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert os.path.exists (ImageQuantizerConfig.srcPath)
        os.makedirs (ImageQuantizerConfig.destPath, exist_ok=True)
        assert os.path.exists (ImageQuantizerConfig.checkpointPath)

        self.inference_loader = DataloaderLite (ImageQuantizerConfig.B, num_processes=ddp_world_size, process_rank=ddp_rank, data_root=ImageQuantizerConfig.srcPath)
    
    def quantize_images_in_shards (self, device, encoder, quantizer, vqgan):

        shard_index = 0
        shard = []  # List to accumulate tensors
        
            
        total_rows = 0  # Track the total number of rows in the current shard

        while self.inference_loader.isZerothEpoch:
            x = self.inference_loader.next_batch()
            x = x.to(device)
            if shard_index == 0:
                tag = "val"
                ImageQuantizerConfig.shardBatchSize = ImageQuantizerConfig.val_shard_size
            else:
                tag = "train"
                ImageQuantizerConfig.shardBatchSize = ImageQuantizerConfig.train_shard_size

            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    ze = encoder(x)
                    pre_quant_activation = vqgan.pre_quant_conv (ze)
                    vq_loss, encoding_indices_tensor, zq = quantizer(pre_quant_activation)

                    # prepend start token to each example in batch
                    
                    start_token = torch.tensor([ImageQuantizerConfig.START_TOKEN]).unsqueeze(0).repeat(x.size(0), 1).to(device)
                    assert start_token.size(0) == encoding_indices_tensor.size(0)
                    encoding_indices_tensor = torch.cat ((start_token, encoding_indices_tensor), dim=1)
                    del zq, vq_loss, ze

                try:
                    # Append the current tensor (B, 256) to the shard

                    shard.append(encoding_indices_tensor)
                    total_rows += encoding_indices_tensor.shape[0]

                    # If we have enough rows for a shard (>= 800), process and save it
                    if total_rows >= ImageQuantizerConfig.shardBatchSize:
                        # Combine all tensors in the shard
                        shard_tensor = torch.cat(shard, dim=0)  # Combine into one tensor

                        # Save the shard
                        shard_path = os.path.join(ImageQuantizerConfig.destPath, f"latent_shard_{tag}_{shard_index:04d}.npy")
                        np.save(shard_path, shard_tensor.cpu().numpy())
                        print(f"Saved shard: latent_shard_{tag}_{shard_index:04d}")

                        # Increment the shard index and reset shard
                        shard_index += 1
                        shard = []  # Clear the list
                        total_rows = 0  # Reset the row counter
                except Exception as e:
                    print(f"Error processing shard: {e}")

            # Handle leftover tensors after the loop ends
        if len(shard) > 0:
            # Combine remaining tensors
            shard_tensor = torch.cat(shard, dim=0)  # Combine into one tensor
            # Save whatever is left (even if less than 800 rows)
            shard_path = os.path.join(ImageQuantizerConfig.destPath, f"latent_shard_{tag}_{shard_index:04d}.npy")
            np.save(shard_path, shard_tensor.cpu().numpy())
            print(f"Final latent shard saved with {shard_tensor.shape[0]:04d} images.")


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup Distributed Data Parallel (DDP)
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE

ddp = int (os.environ.get ('RANK', -1)) != -1 # is this a ddp run or not?
if ddp:
    assert torch.cuda.is_available(), f"CUDA is required for DDP"
    init_process_group (backend='nccl')
    ddp_rank = int (os.environ['RANK'])
    ddp_local_rank = int (os.environ['LOCAL_RANK'])
    ddp_world_size = int (os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device (device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc
else:
    # non ddp vanilla run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr (torch.backends, 'mps') and torch.backends.mps.is_available ():
        device = 'mps'
    print (f"using device: {device}")


torch.manual_seed (1337)
if torch.cuda.is_available ():
    torch.cuda.manual_seed(1337)



from resnet_vqgan import VQGan

vqgan = VQGan ().eval ()
vqgan.to(device)

image_quantizer = ImageQuantizer (ImageQuantizerConfig)

if ddp:
    vqgan = DDP (vqgan,  device_ids=[ddp_local_rank])

raw_vqgan = vqgan.module if ddp else vqgan

assert os.path.exists (ImageQuantizerConfig.checkpointPath)
os.makedirs (ImageQuantizerConfig.destPath, exist_ok=True)
checkpoint = torch.load (ImageQuantizerConfig.checkpointPath)
raw_vqgan.load_state_dict(checkpoint['vqgan_model'])
print (f"Checkpoint weights loaded")
print ("Starting quantization")
if ddp:
    image_quantizer.quantize_images_in_shards (device, vqgan.module.encoder, vqgan.module.quantizer, vqgan.module)
else: 
    image_quantizer.quantize_images_in_shards (device, vqgan.encoder, vqgan.quantizer, vqgan)
print ("quantization complete")

import sys;sys.exit(0)
print ("Validation tests:")

# 1get the shard, decode its images
#  2 get the shardifiedLatents pass them to decoder

from data_utils import DataUtils, Data_Utils_Config

validation_util = DataUtils (Data_Utils_Config)

shard_in_question = load_tokens ("./NoneOfYourBusiness")
shard_in_question = shard_in_question[:8]

os.makedirs("./NoneOfYourBusiness", exist_ok=True)
validation_util.tensor_to_image (shard_in_question, "NoneOfyourBusiness", "clone")

#./shardifiedLatents/latent_shard_0000.npy

if ddp:
    decoder = vqgan.module.decoder
    quantizer = vqgan.module.quantizer
else:
    decoder = vqgan.decoder
    quantizer = vqgan.quantizer

index_tensors = load_indices (".NoneOfYOurBusiness", for_inference=True)
index_tensors = index_tensors[:8].to(device) # (B, 256)
from quantizer import QuantizerConfig
# (B, 256, 1024)
latent_vectors = quantizer.codebook(index_tensors).view(4, 16, 16, QuantizerConfig.n_embd)
latent_vectors = latent_vectors.permute (0, 3, 1, 2).contiguous()
latent_vectors = vqgan.post_quant_conv(latent_vectors)

print (f"shape of tensor plucked out of codebook is:{latent_vectors.shape}")
print ("forwarding to decoder")

image = decoder (latent_vectors) # B, 3, 256, 256
validation_util.tensor_to_image (image, "Noneofyourbusiness", "neural")

