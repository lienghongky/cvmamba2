import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from itertools import islice
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, denoised, multichannel=True):
    """
    Calculate Structural Similarity Index between the original and denoised images.

    Parameters:
    - original (np.ndarray): The original image array.
    - denoised (np.ndarray): The denoised image array.
    - multichannel (bool): If True, the images are assumed to be multi-channel (e.g., RGB).

    Returns:
    - ssim_index (float): The Structural Similarity Index.

    """
            # Convert images to numpy arrays
    original = np.array(original).astype(np.float32)
    denoised = np.array(denoised).astype(np.float32)
    print("Original shape:", original.shape)
    print("Denoised shape:", denoised.shape)
    ssim_index, _ = ssim(original, denoised,channel_axis = 2, multichannel=multichannel, full=True)
    return ssim_index

def calculate_psnr(original, denoised, max_pixel=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio between the original and denoised images.

    Parameters:
    - original (np.ndarray): The original image array.
    - denoised (np.ndarray): The denoised image array.
    - max_pixel (float): The maximum pixel value in the image. Default is 255.0 for 8-bit images.

    Returns:
    - psnr (float): The Peak Signal-to-Noise Ratio.
    """
        # Convert images to numpy arrays
    original = np.array(original).astype(np.float32)
    denoised = np.array(denoised).astype(np.float32)
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')  # No error, PSNR is infinite

    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def pad_reflection(image_array, pad_height, pad_width):
    """
    Pads the image with reflection padding.

    Parameters:
    - image_array (np.ndarray): The image array to pad.
    - pad_height (int): The total padding height needed.
    - pad_width (int): The total padding width needed.

    Returns:
    - padded_image (np.ndarray): The padded image array.
    """
    # Compute the padding for each dimension
    pad_top = 0
    pad_bottom = pad_height 
    pad_left =0
    pad_right = pad_width

    # Apply reflection padding
    padded_image = np.pad(
        image_array,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='reflect'
    )
    return padded_image

def crop_image_to_patches(image_path, patch_size):
    """
    Crops an image into smaller patches with reflection padding if necessary.

    Parameters:
    - image_path (str): The file path to the image.
    - patch_size (tuple): The size (height, width) of each patch.

    Returns:
    - patches (np.ndarray): An array of image patches.
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    
    # Get the dimensions of the image
    height, width, channels = image_array.shape
    patch_height, patch_width = patch_size

    # Calculate padding if necessary
    pad_height = (patch_height - height % patch_height) % patch_height
    pad_width = (patch_width - width % patch_width) % patch_width

    # Pad the image if needed
    if pad_height > 0 or pad_width > 0:
        image_array = pad_reflection(image_array, pad_height, pad_width)

    # Update dimensions after padding
    padded_height, padded_width = image_array.shape[:2]

    # Calculate the number of patches along the height and width
    num_patches_y = padded_height // patch_height
    num_patches_x = padded_width // patch_width

    # Initialize a list to hold the patches
    patches = []

    # Extract patches
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            patch = image_array[
                i * patch_height:(i + 1) * patch_height,
                j * patch_width:(j + 1) * patch_width,
                :
            ]
            patches.append(patch)
    image_array = image_array.transpose(2, 0, 1)
    # Convert the list of patches to a NumPy array
    return np.array(patches), image, image_array

def combine_patches_to_image(patches, image_shape,original_image_shape, patch_size):
    """
    Reconstructs an image from patches.

    Parameters:
    - patches (np.ndarray): An array of image patches.
    - image_shape (tuple): The shape (height, width, channels) of the original image.
    - patch_size (tuple): The size (height, width) of each patch.

    Returns:
    - reconstructed_image (PIL.Image): The reconstructed image.
    """
    print(" image shape:", image_shape) 
    print("original image shape:", original_image_shape)

    # Extract dimensions
    patch_height, patch_width = patch_size
    channels, height, width= image_shape

    # Calculate number of patches in y and x directions
    num_patches_y = height // patch_height
    num_patches_x = width // patch_width

    # Initialize an empty image array
    reconstructed_image_array = np.zeros((height, width, channels), dtype=np.float64)

    # Initialize a count array to keep track of how many patches contribute to each pixel
    count_array = np.zeros((height, width, channels), dtype=np.float64)

    # Place each patch into the reconstructed image
    print("Number of patches_y:", num_patches_y)
    print("Number of patches_x:", num_patches_x)
    print("Number of patches:", patches.shape)
    patch_index = 0
    i=0
    for i in range(num_patches_y):
        j=0
        for j in range(num_patches_x):

            patch = patches[patch_index]
            patch_index += 1
            # Calculate the position to place the patch
            y_start = i * patch_height
            y_end = y_start + patch_height
            x_start = j * patch_width
            x_end = x_start + patch_width
            print(f'patch {patch_index} added to image size {patch.shape} at position {y_start}:{y_end},{x_start}:{x_end}')


            # Add patch to the reconstructed image
            reconstructed_image_array[y_start:y_end, x_start:x_end] += patch
            count_array[y_start:y_end, x_start:x_end] += 1
           

    # Normalize the reconstructed image by dividing by the count array
    reconstructed_image_array = np.divide(reconstructed_image_array, count_array, out=np.zeros_like(reconstructed_image_array), where=count_array!=0)
       # Convert the array to uint8
    reconstructed_image_array = np.clip(reconstructed_image_array, 0, 255).astype(np.uint8)

    print("Reconstructed image shape:", reconstructed_image_array.shape)
    # Ensure the correct order of dimensions for PIL
    if reconstructed_image_array.ndim == 3 and reconstructed_image_array.shape[0] == 3:
        reconstructed_image_array = np.transpose(reconstructed_image_array, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
      # Remove padding by cropping to the original image size
    _, original_height, original_width = original_image_shape
    reconstructed_image_array = reconstructed_image_array[:original_height, :original_width]
    # Convert the array to an image
    reconstructed_image = Image.fromarray(reconstructed_image_array)

    return reconstructed_image

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def postprocess_image(image_tensor):
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy()  # Convert to numpy array
    image = (image * 255).astype('uint8')  # Convert to uint8
    return image

def denoise_image(model, image_path):
    patch_size = (256, 256)  # Define the desired patch size
    patches,image,padded_image = crop_image_to_patches(image_path, patch_size)
    print("Number of patches:", patches.shape[0])
    print("Shape of each patch:", patches.shape[1:])

    denoised_patches = []
    batch_size = 32
    it  = iter(patches)
    for i,chuk_patch in enumerate(iter(lambda: list(islice(it, batch_size)), [])):
        # Convert the list of patches to a tensor
        tensor = torch.tensor(chuk_patch).permute(0, 3, 1, 2).float()/255.0
        # Move tensor to GPU
        tensor = tensor.cuda()

        print(f'Procesing patch {(i+1)*batch_size}/{len(patches)}')
        with torch.no_grad():
            output = model(tensor)
        for patch in output:
            denoised_image = postprocess_image(patch)
            denoised_patches.append(denoised_image)
    # Convert the image to a tensor
    denoised_patches = np.array(denoised_patches)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    print("Original image shape:", image_tensor.shape)

    return combine_patches_to_image(denoised_patches,padded_image.shape,image_tensor.shape,patch_size)

def main(rank, world_size, image_path):
    setup(rank, world_size)
    
    # Load the model
    model = torch.load('models/arc_model.pth')
    
    # Check if model was wrapped in DistributedDataParallel
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    denoised_image = denoise_image(model, image_path)
        # save input image
    image = Image.open(image_path)
    image.save('debug/input.png')
    # save ground truth image
    image = Image.open(image_path.replace('input_crops','target_crops'))
    image.save('debug/image.png')
    # Save the denoised image
    denoised_image.save('debug/denoised_image.png')
    # calculate psnr and ssim
    
    psnr = calculate_psnr(image, denoised_image)
    print(f'PSNR: {psnr:.2f} dB')
    ssim = calculate_ssim(image, denoised_image)
    
    print(f'SSIM: {ssim:.4f}')
    cleanup()

if __name__ == "__main__":
                 #SIDD/target_crops/0200_010_GP_01600_03200_5500_N_SRGB_010.PNG
    image_path = 'SIDD/input_crops/0200_010_GP_01600_03200_5500_N_SRGB_010.PNG'
    # image_path = 'image.mat'


    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main,
                                args=(world_size, image_path),
                                nprocs=world_size,
                                join=True)