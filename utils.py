from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.transforms import Pad
import cv2
from torchvision.transforms import RandomRotation
import torchvision.transforms.functional as F
import torchvision
import random
import csv
from scipy.ndimage import distance_transform_edt
import numpy as np
from skimage.morphology import skeletonize
import shutil
from scipy.ndimage import distance_transform_edt
import time

# class RandomRotationTransform:
#     def __init__(self, degrees, resample=False, expand=False, center=None, fillcolor=0):
#         self.degrees = (-degrees, degrees) if isinstance(degrees, int) else degrees
#         self.resample = resample
#         self.expand = expand
#         self.center = center
#         self.fillcolor = fillcolor

#     def __call__(self, img1, img2):
#         angle = transforms.RandomRotation.get_params(self.degrees)
#         return F.rotate(img1, angle, self.resample, self.expand, self.center, self.fillcolor), \
#                F.rotate(img2, angle, self.resample, self.expand, self.center, self.fillcolor)


def create_csv(csv_file_path):
    with open(csv_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Training Accuracy", "Validation Accuracys", "Best Val DSC", "Learning Rate", "Model Name", "Gamma", "Train Precision", "Train Recall", "Train AUC", "Train mIoU", "Val Precision", "Val Recall", "Val AUC", "Val mIoU", "Train Dice", "Val Dice",  "Epoch Time"])


def total_variation_loss(predicted_veins):
    # Calculate the differences between neighboring pixels along the x and y axes
    diff_x = torch.abs(predicted_veins[:, :, :, 1:] - predicted_veins[:, :, :, :-1])
    diff_y = torch.abs(predicted_veins[:, :, 1:, :] - predicted_veins[:, :, :-1, :])

    # Sum the differences to compute the total variation loss
    loss = torch.sum(diff_x) + torch.sum(diff_y)
    return loss


def tubey_loss(predicted_veins):
    # Binarize the predicted segmentation
    predicted_veins_binary = (predicted_veins > 0.5).float()

    # Apply morphological erosion to encourage thin structures
    kernel = torch.ones(1, 1, 3, 3, device=predicted_veins.device)
    eroded_veins = nn.functional.conv2d(predicted_veins_binary, kernel, padding=1) >= 8

    # Calculate the difference between the eroded and original segmentation
    difference = torch.abs(predicted_veins_binary - eroded_veins.float())

    # Sum the differences to compute the loss
    loss = torch.sum(difference)//predicted_veins.shape[0]
    return loss

class RandomRotationTransform:
    def __init__(self, degrees, fillcolor=0):
        self.rotation = RandomRotation(degrees, fill=fillcolor).to('cuda')

    def __call__(self, img1, img2):
        seed = torch.seed()
        torch.manual_seed(seed)
        if (random.random()>0.5):
            img1 = self.rotation(img1)
            torch.manual_seed(seed)
            img2 = self.rotation(img2)
        return img1, img2
    
    
class RandomTransform:
    def __init__(self, degrees, resample=False, expand=False, center=None, fillcolor=0.5, shift_range=40):
        self.degrees = (-degrees, degrees) if isinstance(degrees, int) else degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fillcolor = fillcolor
        self.shift_range = shift_range  # Max number of pixels to shift in either direction

    def __call__(self, img1, img2):
        angle = transforms.RandomRotation.get_params(self.degrees)
        
        _, height, width = img1.shape
        # If center is not provided, compute image center
        if self.center is None:
            center_x = width // 2
            center_y = height // 2
        else:
            center_x, center_y = self.center
        
        # Add or subtract random values to the center
        shift_x = random.randint(-self.shift_range, self.shift_range)
        shift_y = random.randint(-self.shift_range, self.shift_range)
        modified_center = (center_x + shift_x, center_y + shift_y)
        if self.fillcolor:
            self.fillcolor = random.random()
        return F.rotate(img1, angle, self.resample, self.expand, modified_center, self.fillcolor), \
               F.rotate(img2, angle, self.resample, self.expand, modified_center, 0)

                              
def add_gaussian_noise(image, is_tensor=False, mean=0., std=0.01):
    if not is_tensor:
        tensor = transforms.ToTensor()(image)  # Convert PIL image to Tensor
        noise = torch.randn(tensor.size()).to(tensor.device) * std + mean
        noisy_img = tensor + noise
        noisy_img = transforms.ToPILImage()(noisy_img.clamp(0, 1))  # Convert back to PIL Image
    else:
        noise = torch.randn(image.size(), device=image.device) * std + mean
        noisy_img = image + noise
    return noisy_img

def calculate_l1_norm(model):
    return sum(p.abs().sum() for p in model.parameters())

def add_shade_box(tensor, box_size=(64, 64), shade=0.3):
    """
    Replace a random section in the input tensor with a specified shade.

    Args:
        tensor (torch.Tensor): Input tensor. Shape is (1, 128, 128).
        box_size (tuple of int): Size of the shade box. Default is (32, 32).
        shade (float): Value between 0 and 1 indicating the shade of the box. Default is 0.5.

    Returns:
        torch.Tensor: Tensor with a shade box inserted.
    """
    assert tensor.dim() == 3, "Expected tensor to have 3 dimensions, got {}".format(tensor.dim())
    height, width = tensor.shape[1:]
    assert height >= box_size[0] and width >= box_size[1], "Shade box size exceeds image dimensions."

    top = torch.randint(0, height - box_size[0] + 1, size=(1,)).item()
    left = torch.randint(0, width - box_size[1] + 1, size=(1,)).item()

    tensor[:, top:top+box_size[0], left:left+box_size[1]] *= shade
    return tensor


def add_black_box(tensor, box_size=(32, 32)):
    """
    Replace a random section in the input tensor with zeroes (a black box).

    Args:
        tensor (torch.Tensor): Input tensor. Shape is (1, 128, 128).
        box_size (tuple of int): Size of the black box. Default is (32, 32).

    Returns:
        torch.Tensor: Tensor with a black box inserted.
    """
    assert tensor.dim() == 3, "Expected tensor to have 3 dimensions, got {}".format(tensor.dim())
    height, width = tensor.shape[1:]
    assert height >= box_size[0] and width >= box_size[1], "Black box size exceeds image dimensions."

    top = torch.randint(0, height - box_size[0] + 1, size=(1,)).item()
    left = torch.randint(0, width - box_size[1] + 1, size=(1,)).item()

    tensor[:, top:top+box_size[0], left:left+box_size[1]] = 0
    return tensor

def save_nonsigmoid_image(tensor, filename):
    # tensor.clamp_(0, 1)
    torch.sigmoid(tensor)
    save_image(tensor, filename)
    
def save_sigmoid_image(tensor, filename):
    tensor.clamp_(0, 1)
    save_image(tensor, filename)

def save_normalized_image(tensor, filename):
    # Denormalize the generated images
    denormalized_tensor = (tensor + 1.) / 2  # Convert from [-1, 1] to [0, 1]

    denormalized_tensor.clamp_(0, 1)
    save_image(denormalized_tensor, filename)


# class DiceLoss(torch.nn.Module):
#     def __init__(self, eps=1e-7):
#         super(DiceLoss, self).__init__()
#         self.eps = eps
        
#     def forward(self, output, target):
#         intersection = (output * target).sum(dim=(2,3))
#         union = output.sum(dim=(2,3)) + target.sum(dim=(2,3))

#         # A 'smooth' term (usually 1) is added to the numerator and denominator to prevent zero division
#         dice = (2. * intersection + self.eps) / (union + self.eps)

#         # 'clamp' is used to ensure that dice coefficient stays between 0 and 1
#         dice = torch.clamp(dice, 0, 1)

#         return 1. - dice.mean()

# class TanhNormalize(object):
#     def __call__(self, tensor):
#         return tensor * 2. - 1.

# Modified DiceLoss class with additional checks and enhanced numerical stability
class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        # Check for NaN and Inf values in output and target
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Output contains NaN or Inf values.")
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("Target contains NaN or Inf values.")
        
        # Clamping to avoid extreme values
        output = torch.clamp(output, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Compute Intersection and Union
        intersection = (output * target).sum(dim=(2, 3))
        union = output.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        # Compute Dice coefficient
        dice = (2. * intersection) / (union + self.eps)
        
        # Clamping the Dice coefficient between 0 and 1
        dice = torch.clamp(dice, 0, 1)
        
        return 1. - dice.mean()


class SoftClDiceLoss(nn.Module):
    def __init__(self, centerline_weight=2.0):
        super(SoftClDiceLoss, self).__init__()
        self.centerline_weight = centerline_weight

    def forward(self, input, target):
        # Ensure the tensors are [N, C, H, W]
        if len(input.shape) == 3:
            input = input.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)

        # Calculate Dice Coefficient
        intersection = (input * target).sum(dim=(2, 3))
        union = input.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
        
        # Create centerline mask
        centerline = torch.ones_like(target)
        centerline[:,:, target.shape[2]//4: 3 * target.shape[2]//4, target.shape[3]//4: 3 * target.shape[3]//4] = self.centerline_weight

        # Calculate centerline weighted Dice Loss
        centerline_intersection = (input * target * centerline).sum(dim=(2, 3))
        centerline_union = (input * centerline).sum(dim=(2, 3)) + (target * centerline).sum(dim=(2, 3))

        centerline_dice = (2 * centerline_intersection + 1e-6) / (centerline_union + 1e-6)
        
        # Combine regular dice and centerline weighted dice
        combined_dice = (dice_score + centerline_dice) / 2.0

        # Return 1 - Dice to minimize loss
        return 1 - combined_dice.mean()

# def compute_dtm(img):
#     img_cpu = img.cpu().numpy()
#     dtm = distance_transform_edt(img_cpu)
#     return torch.tensor(dtm, dtype=torch.float32, device=img.device)

# def WNBL(pred, target):
#     dtm = compute_dtm(target)
#     loss = 1 - torch.sum((dtm * (1 - target)**2 - pred)**2) / torch.sum((dtm * (1 - target)**2 - target)**2)
#     return loss
def compute_dtm(img):
    img_cpu = img.cpu().numpy()
    dtm = distance_transform_edt(img_cpu)
    return torch.tensor(dtm, dtype=torch.float32, device=img.device)

def compute_class_weights(mask):
    class_weights = []
    total_voxels = mask.numel()
    for k in torch.unique(mask):
        wk = 1 / (torch.sum(mask == k).float() / total_voxels).item()**2
        class_weights.append(wk)
    return torch.tensor(class_weights, dtype=torch.float32, device=mask.device)

def WNBL(pred, target):
    dtm = compute_dtm(target)
    class_weights = compute_class_weights(target)
    
    loss = torch.tensor(0.0, device=pred.device)
    for k, wk in enumerate(class_weights):
        mask_k = (target == k).float()
        pred_k = (pred == k).float()
        
        term1 = torch.sum((dtm * (1 - mask_k)**2 - pred_k)**2)
        term2 = torch.sum((dtm * (1 - mask_k)**2 - mask_k)**2)
        
        loss += wk * term1 / term2
    
    loss = 1 - loss / len(class_weights)
    return loss


class WeightedNormalizedBoundaryLoss(nn.Module):
    def __init__(self):
        super(WeightedNormalizedBoundaryLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: Tensor of shape [batch_size, num_classes, H, W] (predictions)
        target: Tensor of shape [batch_size, num_classes, H, W] (ground truth)
        distance_map: Tensor of shape [batch_size, num_classes, H, W] (distance transform map)
        class_weights: Tensor of shape [num_classes] (class weights)
        """
        distance_map = chamfer_distance_map(target)
        class_weights = compute_class_weights(target)

        # Calculate (1 - target)^2
        target_comp_sq = (1 - target).pow_(2)

        # Calculate the numerator
        num = (distance_map * (target_comp_sq - pred).pow_(2)).sum(dim=(2, 3))
        num = (class_weights * num).sum(dim=1)

        # Calculate the denominator
        den = (distance_map * (target_comp_sq - target).pow_(2)).sum(dim=(2, 3))
        den = (class_weights * den).sum(dim=1)

        # Calculate the loss
        wnbl_loss = 1 - (num / den)
        wnbl_loss = wnbl_loss.mean()

        return wnbl_loss

def chamfer_distance_map(binary_image):
    # Create tensors to hold vertical and horizontal distance
    vertical_dist = torch.zeros_like(binary_image)
    horizontal_dist = torch.zeros_like(binary_image)

    # Compute vertical distance
    vertical_dist[:, :, 1:, :] = binary_image[:, :, :-1, :]  # Shift down
    vertical_dist -= binary_image
    vertical_dist = vertical_dist.cumsum(dim=-2)
    
    # Compute backward cumsum for vertical distance
    vertical_dist = vertical_dist.flip(dims=[-2])
    vertical_dist = vertical_dist.cumsum(dim=-2)
    vertical_dist = vertical_dist.flip(dims=[-2])

    # Compute horizontal distance (similarly)
    horizontal_dist[:, :, :, 1:] = binary_image[:, :, :, :-1]  # Shift right
    horizontal_dist -= binary_image
    horizontal_dist = horizontal_dist.cumsum(dim=-1)

    # Compute backward cumsum for horizontal distance
    horizontal_dist = horizontal_dist.flip(dims=[-1])
    horizontal_dist = horizontal_dist.cumsum(dim=-1)
    horizontal_dist = horizontal_dist.flip(dims=[-1])

    # Combine to form distance map
    distance_map = torch.sqrt(vertical_dist ** 2 + horizontal_dist ** 2)

    return distance_map

# class WeightedNormalizedBoundaryLoss(nn.Module):
#     def __init__(self):
#         super(WeightedNormalizedBoundaryLoss, self).__init__()

#     def forward(self, pred, target):
#         """
#         pred: Tensor of shape [batch_size, num_classes, H, W] (predictions)
#         target: Tensor of shape [batch_size, num_classes, H, W] (ground truth)
#         """
#         start_time = time.time()
#         # distance_map = compute_dtm(target)  # Assuming you have a function to compute this
#         distance_map = chamfer_distance_map(target)
#         print(f"Time taken for distance_map computation: {time.time() - start_time:.4f} seconds")

#         start_time = time.time()
#         class_weights = compute_class_weights(target)  # Assuming you have a function to compute this
#         print(f"Time taken for class_weights computation: {time.time() - start_time:.4f} seconds")

#         start_time = time.time()
#         # Calculate (1 - target)^2
#         target_comp_sq = (1 - target).pow_(2)
#         print(f"Time taken for target_comp_sq computation: {time.time() - start_time:.4f} seconds")

#         start_time = time.time()
#         # Calculate the numerator
#         num = (distance_map * (target_comp_sq - pred).pow_(2)).sum(dim=(2, 3))
#         num = (class_weights * num).sum(dim=1)
#         print(f"Time taken for numerator computation: {time.time() - start_time:.4f} seconds")

#         start_time = time.time()
#         # Calculate the denominator
#         den = (distance_map * (target_comp_sq - target).pow_(2)).sum(dim=(2, 3))
#         den = (class_weights * den).sum(dim=1)
#         print(f"Time taken for denominator computation: {time.time() - start_time:.4f} seconds")

#         start_time = time.time()
#         # Calculate the loss
#         wnbl_loss = 1 - (num / den)
#         wnbl_loss = wnbl_loss.mean()
#         print(f"Time taken for wnbl_loss computation: {time.time() - start_time:.4f} seconds")

#         return wnbl_loss

def enhance_veins(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply CLAHE to the grayscale image to improve contrast in local areas
    clahe_img = clahe.apply(gray)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Apply Gamma correction with a gamma of 2.0
    gamma_corrected = cv2.pow(min_max_contrast/255., 2.0)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    return gamma_corrected


# def enhance_veins_v4(arm):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

#     # Equalize the histogram of the grayscale image to standardize contrast and brightness
#     equalized = cv2.equalizeHist(gray)

#     # Gaussian blur to reduce the appearance of hair
#     blur = cv2.GaussianBlur(equalized, (5, 5), 0)

#     # Morphological dilation to further reduce the appearance of hair
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#     morphed = cv2.dilate(blur, kernel)

#     # Create a CLAHE object (Arguments are optional)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

#     # Apply CLAHE to the grayscale image to improve contrast in local areas
#     clahe_img = clahe.apply(morphed)

#     # Apply a median filter to remove noise
#     filtered_image = cv2.medianBlur(clahe_img, 5)

#     # Apply contrast stretching
#     min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

#     # Lower the gamma value to reduce the blurring
#     gamma = 1.5

#     # Apply Gamma correction with a reduced gamma
#     gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

#     # Convert the float64 image to uint8
#     gamma_corrected = gamma_corrected.astype(np.uint8)
    
#     # Apply the black hat morphological operation with a smaller structuring element
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

#     # Subtract the black hat image from the gamma corrected image to better extract the veins
#     enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
#     return enhanced_veins

# Updated function to enhance veins without Gaussian blur and morphological dilation
def enhance_veins_v4(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # # Equalize the histogram of the grayscale image to standardize contrast and brightness
    # equalized = cv2.equalizeHist(gray)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # Apply CLAHE to the grayscale image to improve contrast in local areas
    clahe_img = clahe.apply(gray)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Lower the gamma value to reduce the blurring
    gamma = 1.5

    # Apply Gamma correction with a reduced gamma
    gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Apply the black hat morphological operation with a smaller structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

    # Subtract the black hat image from the gamma corrected image to better extract the veins
    enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
    return enhanced_veins

def enhance_veins_v4_color(arm):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(arm, cv2.COLOR_RGB2HSV)
    
    # Split the image into its color channels
    h, s, v = cv2.split(hsv)
    
    # Equalize the histogram of the V channel to standardize contrast and brightness
    equalized = cv2.equalizeHist(v)

    # Gaussian blur to reduce the appearance of hair
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Morphological dilation to further reduce the appearance of hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    morphed = cv2.dilate(blur, kernel)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # Apply CLAHE to the V channel to improve contrast in local areas
    clahe_img = clahe.apply(morphed)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Lower the gamma value to reduce the blurring
    gamma = 1.5

    # Apply Gamma correction with a reduced gamma
    gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Apply the black hat morphological operation with a smaller structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

    # Subtract the black hat image from the gamma corrected image to better extract the veins
    enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
    # Merge the processed V channel back with the original H and S channels
    hsv_enhanced = cv2.merge([h, s, enhanced_veins])
    
    # Convert the image back to RGB color space
    output = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return output


def enhance_veins_v5(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # Apply CLAHE to the grayscale image to improve contrast in local areas
    clahe_img = clahe.apply(gray)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Lower the gamma value to reduce the blurring
    gamma = 1.5

    # Apply Gamma correction with a reduced gamma
    gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Apply the black hat morphological operation with a smaller structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

    # Subtract the black hat image from the gamma corrected image to better extract the veins
    enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
    # Apply adaptive thresholding to the subtracted image
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_veins, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return adaptive_thresh

def enhance_veins_v6(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram of the grayscale image to standardize contrast and brightness
    equalized = cv2.equalizeHist(gray)

    # Gaussian blur to reduce the appearance of hair
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Morphological dilation to further reduce the appearance of hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    morphed = cv2.dilate(blur, kernel)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply CLAHE to the morphed image to improve contrast in local areas
    clahe_img = clahe.apply(morphed)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Lower the gamma value to reduce the blurring
    gamma = 1.5

    # Apply Gamma correction with a reduced gamma
    gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Apply the black hat morphological operation with a smaller structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

    # Subtract the black hat image from the gamma corrected image to better extract the veins
    enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
    # Apply adaptive thresholding to the subtracted image
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_veins, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return adaptive_thresh

def enhance_veins_v7(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram of the grayscale image to standardize contrast and brightness
    equalized = cv2.equalizeHist(gray)

    # Gaussian blur to reduce the appearance of hair
    blur = cv2.GaussianBlur(equalized, (5, 5), 0)

    # Morphological dilation to further reduce the appearance of hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    morphed = cv2.dilate(blur, kernel)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply CLAHE to the morphed image to improve contrast in local areas
    clahe_img = clahe.apply(morphed)

    # Apply a median filter to remove noise
    filtered_image = cv2.medianBlur(clahe_img, 5)

    # Apply contrast stretching
    min_max_contrast = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Lower the gamma value to reduce the blurring
    gamma = 1.5

    # Apply Gamma correction with a reduced gamma
    gamma_corrected = cv2.pow(min_max_contrast/255., gamma)*255

    # Convert the float64 image to uint8
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Apply the black hat morphological operation with a smaller structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_BLACKHAT, kernel)

    # Subtract the black hat image from the gamma corrected image to better extract the veins
    enhanced_veins = cv2.subtract(gamma_corrected, blackhat)
    
    # Apply adaptive thresholding to the subtracted image with increased block size and slightly higher C (constant subtracted from the mean)
    adaptive_thresh = cv2.adaptiveThreshold(enhanced_veins, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    
    return adaptive_thresh


def enhance_arm_fast_otsu(arm):
    # # Convert the image to grayscale
    # gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # # Gaussian blur to reduce noise and smooth the image
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # # Apply Otsu's thresholding to the blurred image
    # _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # return otsu_thresh
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise and smooth the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply manual thresholding to the blurred image
    _, manual_thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)
    
    return manual_thresh


def enhance_arm_fast_threshold(arm):
    # Convert the image to grayscale
    gray = cv2.cvtColor(arm, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return adaptive_thresh


class PairedImageDatasetArm(Dataset):
    def __init__(self, arms_dir, masks_dir, transform=None):
        self.arms_dir = arms_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.masks_list = sorted(os.listdir(masks_dir))
        self.arms_list = sorted(os.listdir(arms_dir))

    def __len__(self):
        return min(len(self.masks_list), len(self.arms_list))

    def __getitem__(self, idx):
        masks_path = os.path.join(self.masks_dir, self.masks_list[idx])
        arms_path = os.path.join(self.arms_dir, self.arms_list[idx])
        mask = Image.open(masks_path).convert("RGB")
        arm = cv2.imread(arms_path)
        arm = enhance_veins_v4(arm)
        arm = Image.fromarray(arm)
        if self.transform is not None:
            mask = self.transform(mask)
            arm = self.transform(arm)
            # Binarize the mask image
            mask = (mask > 0.5).float()
        return arm, mask


class PairedImageDataset(Dataset):
    def __init__(self, arms_dir, masks_dir, transform=None):
        self.arms_dir = arms_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.masks_list = sorted(os.listdir(masks_dir))
        self.arms_list = sorted(os.listdir(arms_dir))

    def __len__(self):
        return min(len(self.masks_list), len(self.arms_list))

    def __getitem__(self, idx):
        masks_path = os.path.join(self.masks_dir, self.masks_list[idx])
        arms_path = os.path.join(self.arms_dir, self.arms_list[idx])
        mask = Image.open(masks_path).convert("L")
        arm = Image.open(arms_path).convert("L")
        if self.transform is not None:
            mask = self.transform(mask)
            arm = self.transform(arm)
            # Binarize the mask image
            mask = (mask > 0.5).float()
        return arm, mask


class PairedImageDatasetEnhanced(Dataset):
    def __init__(self, arms_dir, masks_dir, transform=None):
        self.arms_dir = arms_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.masks_list = sorted(os.listdir(masks_dir))
        self.arms_list = sorted(os.listdir(arms_dir))
        
    def __len__(self):
        return min(len(self.masks_list), len(self.arms_list))

    def __getitem__(self, idx):
        masks_path = os.path.join(self.masks_dir, self.masks_list[idx])
        arms_path = os.path.join(self.arms_dir, self.arms_list[idx])
        mask = Image.open(masks_path).convert("L")
        arm_img = Image.open(arms_path).convert("RGB")
        
        arm = cv2.imread(arms_path)
        enhanced_arm = enhance_veins_v4(arm)
        enhanced_arm_pil = Image.fromarray(enhanced_arm)
        # arm = Image.open(arms_path).convert("RGB")
        if self.transform is not None:
            mask = self.transform(mask)
            enhanced_arm_pil = self.transform(enhanced_arm_pil)
            arm_img = self.transform(arm_img)
            # arm = self.transform(arm)
            # Binarize the mask image
            mask = (mask > 0.5).float()

        # return torch.cat((arm_img, enhanced_arm_pil), 0), mask  # concat the enhanced arm to the channels
        return enhanced_arm_pil, mask


class PairedImageDatasetEnhancedWithArm(Dataset):
    def __init__(self, arms_dir, masks_dir, transform=None):
        self.arms_dir = arms_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.masks_list = sorted(os.listdir(masks_dir))
        self.arms_list = sorted(os.listdir(arms_dir))
        
    def __len__(self):
        return min(len(self.masks_list), len(self.arms_list))

    def __getitem__(self, idx):
        masks_path = os.path.join(self.masks_dir, self.masks_list[idx])
        arms_path = os.path.join(self.arms_dir, self.arms_list[idx])
        mask = Image.open(masks_path).convert("L")        
        arm = cv2.imread(arms_path)
        enhanced_arm = enhance_arm_fast_threshold(arm)
        enhanced_veins = enhance_veins_v4(arm)
        enhanced_veins_pil = Image.fromarray(enhanced_veins)
        enhanced_arm_pil = Image.fromarray(enhanced_arm)
        # arm = Image.open(arms_path).convert("RGB")
        if self.transform is not None:
            mask = self.transform(mask)
            enhanced_veins_pil = self.transform(enhanced_veins_pil)
            enhanced_arm_pil = self.transform(enhanced_arm_pil)
            # Binarize the mask image
            mask = (mask > 0.5).float()

        return torch.cat((enhanced_arm_pil, enhanced_veins_pil), 0), mask  # concat the enhanced arm to the channels


class PairedTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        img1 = self.transforms(img1)
        random.seed(seed)
        img2 = self.transforms(img2)
        return img1, img2


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        
        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)  # Softmax applied to the last dimension

    def forward(self, input):
        # Extract batch_size and dimension
        batch_size, C, width, height = input.size()

        # Apply convolutions
        query = self.query_conv(input).view(batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(input).view(batch_size, -1, width*height)
        value = self.value_conv(input).view(batch_size, -1, width*height)
        
        # Calculate attention
        attention = self.softmax(torch.bmm(query, key))
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Apply gamma and add original input
        out = self.gamma*out + input

        # return out, attention
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        # Convolutional layer to produce the attention map
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Sigmoid activation to produce attention scores between 0 and 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate the attention map
        attention_map = self.conv1(x)
        
        # Apply sigmoid activation
        attention_map = self.sigmoid(attention_map)
        
        # Apply the attention map to the input tensor
        out = x * attention_map
        
        return out


def get_last_completed_epoch(csv_file_path):
    last_epoch = 0
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r") as f:
            reader = csv.reader(f)
            last_row = None
            for row in reader:
                if row.__len__() != 0:
                    last_row = row
            if last_row:
                try:
                    last_epoch = int(last_row[0]) + 1
                except:
                    last_epoch = 0
    return last_epoch



def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
