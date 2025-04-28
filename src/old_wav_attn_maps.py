import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from pytorch_wavelets import DWTForward, DWTInverse
xfm = DWTForward(J=1).to(device)
ifm = DWTInverse().to(device)

import requests
from io import BytesIO
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import pywt
import numpy as np
from skimage.transform import resize

def init_and_get_model(patch_size = 8):

    #model = torch.hub.load('facebookresearch/dino:main', 'dino_deits8')
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for p in model.parameters():
            p.requires_grad = False

    model.eval()
    model.to(device)
    return model

def preprocess_image(image, patch_size, transform):
    if transform is None:
        transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    try: # if it is a PIL image 
        img = transform(image)
    except: # if it is a tensor
        img = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    # print(img.shape)
    # make the image divisible by the patch size
    #if it is a batch
    if len(img.shape) == 4:
        w, h = img.shape[2] - img.shape[2] % patch_size, img.shape[3] - img.shape[3] % patch_size
        img = img[:, :, :w, :h]
    else:
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)
        
    # w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    # img = img[:, :w, :h].unsqueeze(0)

    w_featmap_img = img.shape[-2] // patch_size
    h_featmap_img = img.shape[-1] // patch_size
    img = img.to(device)
    return img, w_featmap_img, h_featmap_img
# try:
#     image = Image.open("MarkAlbertazzi-1-4_3museumShips.jpg")
# except:
#     response = requests.get("https://sdmaritime.org/wp-content/uploads/2020/01/MarkAlbertazzi-1-4_3museumShips.jpg")
#     image = Image.open(BytesIO(response.content)).convert("RGB")
#     image.save("MarkAlbertazzi-1-4_3museumShips.jpg")





def fwpt2d_torch(image, wavelet, level):
    """
    Perform a 2D Full Wavelet Packet Transform on an image and convert the results to PyTorch tensors.

    Parameters:
    image (torch.Tensor): Input image.
    wavelet (str): Wavelet to use.
    level (int): Number of decomposition levels.

    Returns:
    list: List of wavelet packet coefficients as PyTorch tensors, upsampled to the original image size.
    """
    # Convert the PyTorch tensor to a NumPy array
    image_np = image.cpu().numpy()

    # Perform the 2D Full Wavelet Packet Transform using PyWavelets
    wp = pywt.WaveletPacket2D(data=image_np, wavelet=wavelet, mode='symmetric', maxlevel=level)
    coeffs_np = [wp[node.path].data for node in wp.get_level(level, 'natural')]

    # Upsample the coefficients to the original image size
    original_size = image_np.shape
    coeffs_upsampled = [resize(coeff, original_size, mode='reflect', anti_aliasing=True) for coeff in coeffs_np]

    # Convert the upsampled coefficients to PyTorch tensors
    coeffs_torch = [torch.tensor(coeff).to(device) for coeff in coeffs_upsampled]
    return coeffs_torch


#wavelet dwt2dfwt
def get_fwpt2d_attention_map(model, img, w_featmap_img, h_featmap_img):
    
    coeffs_torch = fwpt2d_torch(img, wavelet="db1", level=1)

    # LL, Yh = xfm(img)
    LL, LH, HL, HH = coeffs_torch
    # wav_cat = torch.cat([LL,LH,HL,HH], dim=1)

    w_featmap_wav = w_featmap_img
    h_featmap_wav = h_featmap_img


    attentions_wav = [model.get_last_selfattention(LL),
                    model.get_last_selfattention(LH),
                    model.get_last_selfattention(HL),
                    model.get_last_selfattention(HH) ]   #img.cuda()
    return attentions_wav, w_featmap_wav, h_featmap_wav


def get_attn_old(attentions, nh, w_featmap, h_featmap, patch_size):
    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    threshold = 0.6 # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]

    th_attn = th_attn.reshape(nh, w_featmap//2, h_featmap//2).float()

    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = attentions.reshape(nh, w_featmap//2, h_featmap//2)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions_mean = np.mean(attentions, axis=0)
    return attentions, attentions_mean


def get_attn(attentions, nh, w_featmap, h_featmap, patch_size, threshold = 0.6):
    # Sort and normalize attentions
    val, idx = torch.sort(attentions, dim=-1)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    # We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass.
    th_attn = cumval > (1 - threshold)
    
    # Reorder based on the original indices
    idx2 = torch.argsort(idx, dim=-1)
    th_attn = torch.gather(th_attn, dim=-1, index=idx2)
    
    # Reshape and interpolate
    th_attn = th_attn.reshape(-1, nh, w_featmap // 2, h_featmap // 2).float()
    th_attn = nn.functional.interpolate(th_attn, scale_factor=patch_size, mode="nearest")
    
    attentions = attentions.reshape(-1, nh, w_featmap // 2, h_featmap // 2)
    attentions = nn.functional.interpolate(attentions, scale_factor=patch_size, mode="nearest")
    
    # Convert to numpy and calculate mean
    # th_attn = th_attn.cpu().numpy()
    # attentions = attentions.cpu().numpy()
    # attentions_mean = np.mean(attentions, axis=1)
    attentions_mean = torch.mean(attentions, dim=1)
    return attentions, attentions_mean




def time_dependent_masking(A, T, l):
    """
    Generates time-dependent masks for a diffusion process based on an attention map using PyTorch.

    Parameters:
    - A (torch.Tensor): Attention map with values in the range [0, 1] representing semantic importance. Shape: (B, H, W).
    - T (int): Maximum number of diffusion time steps.
    - l (float): Lower bound ensuring a minimum refinement level for each position.

    Returns:
    - masks (torch.Tensor): A tensor of shape (B, T, H, W) where each mask[b, t, i, j] is 1 if the position (i, j)
                             should be refined at time step t for batch b, and 0 otherwise.
    """
    B, H, W = A.shape  # Get the batch size, height, and width of the attention map
    masks = torch.zeros((B, T, H, W), dtype=torch.int)  # Initialize masks for all time steps

    # Calculate the maximum refinement threshold for each spatial position
    thresholds = T * (A + l).unsqueeze(1)  # Shape: (B, 1, H, W)

    # Generate masks based on these thresholds
    t_values = torch.arange(T, dtype=torch.float32, device=A.device).view(1, T, 1, 1)  # Shape: (1, T, 1, 1)
    masks = (thresholds >= (T - t_values)).int()  # Shape: (B, T, H, W)

    return masks

# Example usage:
# B, H, W = 4, 64, 64  # Batch size and dimensions of the attention map
# A = torch.rand(B, H, W)  # Example attention map with values in [0, 1]
# T = 50  # Total number of diffusion steps
# l = 0.2  # Lower bound hyperparameter
# masks = time_dependent_masking(A, T, l)
# print(masks.shape)  # Should print (B, T, H, W)

def get_M_attn_maps(T=50,l=0.2, attentions_mean_img=None, attentions_wav_tuples=None):

    A_img = (attentions_mean_img - attentions_mean_img.min()) / (attentions_mean_img.max() - attentions_mean_img.min())
    # A_img = torch.from_numpy(scaled_tensor_img)
    if attentions_wav_tuples is not None:
        scaled_tensor_wav_L = [(attentions_wav_tuples[i][1] - attentions_wav_tuples[i][1].min()) / (attentions_wav_tuples[i][1].max() - attentions_wav_tuples[i][1].min())
                                for i in range(len(attentions_wav_tuples))]
        A_wav_L = [torch.from_numpy(scaled_tensor_wav) for scaled_tensor_wav in scaled_tensor_wav_L]
        masks_wav_L = [time_dependent_masking(A_wav, T, l) for A_wav in A_wav_L]
    else:
        masks_wav_L = None
    # T = 50  # Total number of diffusion steps
    # l = 0.2  # Lower bound hyperparameter

    masks_img = time_dependent_masking(A_img, T, l)
    return masks_img, masks_wav_L



def get_img_resized_like_attentions_map(img):
    return pth_transforms.Resize((attentions_wav_tuples[0][1].shape[0], attentions_wav_tuples[0][1].shape[1]))(img.squeeze())

def plot_mask_at_specific_timesteps(T, masks_img, masks_wav_L):
    # Plot the masks at specific time steps to visualize the gradual refinement
    time_steps_to_plot = [T - 1, T // 2, T // 3 ,T // 4, 0]

    plt.figure(figsize=(10, 6))
    for i, t in enumerate(time_steps_to_plot):
        plt.subplot(1, len(time_steps_to_plot), i + 1)
        plt.imshow(masks_img[t].cpu().numpy(), cmap='gray')
        plt.title(f'Mask at t={T - t}')
        plt.axis('off')
    plt.suptitle("Time-Dependent Masking")
    plt.show()

    wav_sub_str = ["LL","LH","HL","HH", ]
    for j in range(len(masks_wav_L)):
        plt.figure(figsize=(10, 5))
        for i, t in enumerate(time_steps_to_plot):
            plt.subplot(1, len(time_steps_to_plot), i + 1)
            plt.imshow(masks_wav_L[j][t].cpu().numpy(), cmap='gray')
            plt.title(f'Mask at t={T - t}')
            plt.axis('off')
        plt.suptitle(f"Time-Dependent Masking {wav_sub_str[j]}")
        plt.show()
    plt.savefig("plot_nave.png")



#example usage
if __name__ == "__main__":
    patch_size = 8
    
    
    model = init_and_get_model(patch_size=patch_size)
    image = Image.open("MarkAlbertazzi-1-4_3museumShips.jpg")
    img, w_featmap_img, h_featmap_img = preprocess_image(image, patch_size=patch_size, transform=None)
    attentions_img = model.get_last_selfattention(img)   #img.cuda()
    attentions_wav, w_featmap_wav, h_featmap_wav = get_fwpt2d_attention_map(model, img, w_featmap_img, h_featmap_img)
        
    nh_img = attentions_img.shape[1] # gives the number of heads.
    nh_wav = attentions_wav[0].shape[1] # gives the number of heads.

    # we keep only the output patch attention
    # Selects the attention weights of the first element in the batch (attentions[0])
    # and specifically focuses on the attention of the [CLS] token (index 0 for patch tokens) to all other patches (1:).
    # It reshapes this selection to (nh, -1), where each row corresponds to the attention values of a different head across all spatial positions (flattened).
    # After this, attentions.shape would be (nh, num_patches - 1).
    attentions_img = attentions_img[0, :, 0, 1:].reshape(nh_img, -1)
    attentions_wav = [attentions_wav[i][0, :, 0, 1:].reshape(nh_wav, -1) for i in range(len(attentions_wav))]

    # attentions = attentions.reshape(nh, -1)

    # print(attentions_img.shape)
    # print(attentions_wav[0].shape)
    attentions_img, attentions_mean_img = get_attn_old(attentions_img, nh_img, w_featmap_img, h_featmap_img, patch_size=patch_size)
    attentions_wav_tuples = [get_attn(attentions_wav[i], nh_wav, w_featmap_wav, h_featmap_wav, patch_size=patch_size) for i in range(len(attentions_wav))]
    plot_mask_at_specific_timesteps(50, *get_M_attn_maps(50,0.2, attentions_mean_img=attentions_mean_img, attentions_wav_tuples=attentions_wav_tuples))