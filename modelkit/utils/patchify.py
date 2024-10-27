
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def pad_image(image, patch_size):
    """
    Pads the image to make its dimensions divisible by patch size.
    
    Args:
        image (PIL.Image): The input image.
        patch_size (int): The size of the patches.
        
    Returns:
        PIL.Image: The padded image.
    """
    image = Image.fromarray(image)
    # Calculate padding
    pad_height = (patch_size - image.shape[1] % patch_size) % patch_size
    pad_width = (patch_size - image.shape[0] % patch_size) % patch_size
    
    # Apply padding
    padding = (0, 0, pad_width, pad_height)  # (left, top, right, bottom)
    padded_image = transforms.functional.pad(image, padding)
    
    return padded_image

def create_patches(image_tensor, patch_size, stride):
    """
    Creates patches from the image tensor.
    
    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (1, C, H, W).
        patch_size (int): The size of the patches.
        stride (int): The stride for creating patches.
        
    Returns:
        torch.Tensor: The patches tensor of shape (N, C, patch_size, patch_size).
    """
    # Create patches using unfold
    patches = image_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, patch_size, patch_size)
    return patches

def reconstruct_image_from_patches(patches, image_shape, patch_size, stride):
    """
    Reconstructs the image from patches.
    
    Args:
        patches (torch.Tensor): The patches tensor of shape (N, C, patch_size, patch_size).
        image_shape (tuple): The shape of the original image tensor (1, C, H, W).
        patch_size (int): The size of the patches.
        stride (int): The stride for creating patches.
        
    Returns:
        torch.Tensor: The reconstructed image tensor of shape (1, C, H, W).
    """
    # Calculate the number of patches along height and width
    num_patches_h = image_shape[2] // patch_size
    num_patches_w = image_shape[3] // patch_size
    
    # Reshape patches
    reconstructed_patches = patches.view(1, num_patches_h, num_patches_w, 3, patch_size, patch_size)
    reconstructed_patches = reconstructed_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # Fold back to the original image
    reconstructed_image = reconstructed_patches.view(1, 3, image_shape[2], image_shape[3])
    
    return reconstructed_image

def tensor_to_image(tensor):
    """
    Converts a tensor to a PIL image.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (C, H, W).
        
    Returns:
        PIL.Image: The output image.
    """
    tensor = tensor.permute(1, 2, 0).numpy()
    image = (tensor * 255).astype(np.uint8)
    return Image.fromarray(image)

def image_to_tensor(image):
    """
    Converts a PIL image to a tensor.
    
    Args:
        image (PIL.Image): The input image.
        
    Returns:
        torch.Tensor: The output tensor of shape (1, C, H, W).
    """
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

