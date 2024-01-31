import os
from PIL import Image
from torchvision import transforms
import random
import cv2
from utils import *
from PIL import Image
import numpy as np
from PIL import ImageEnhance

IMG_IN_DIR = './datasets/vein_seg_dataset/images'
SEG_IN_DIR = './datasets/vein_seg_dataset/targets'
IMG_OUT_DIR = './datasets/vein_seg_dataset/images_aug'
SEG_OUT_DIR = './datasets/vein_seg_dataset/targets_aug'

VAL_IMG_IN_DIR = './datasets/vein_seg_dataset/val_images'
VAL_SEG_IN_DIR = './datasets/vein_seg_dataset/val_targets'
VAL_IMG_OUT_DIR = './datasets/vein_seg_dataset/val_images_aug'
VAL_SEG_OUT_DIR = './datasets/vein_seg_dataset/val_targets_aug'


# Custom brightness adjustment
def brightness_adjustment(img, factor=0.5):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def rotation_90(img):
    return img.rotate(90)

def rotation_180(img):
    return img.rotate(180)

    
    
def generate_data(img_path, seg_path):
    # Load the input image
    # Enhance veins in the input image
    img = cv2.imread(img_path)

    enhanced_img = enhance_veins_v4(img)
    img = Image.fromarray(enhanced_img).convert("L")
    seg_img = Image.open(seg_path).convert("L")
    fillcolor = 0.
    # common_transform_list = [
    #     transforms.functional.hflip,
    #     transforms.functional.vflip,
    #     # Random zoom, ensuring that the same crop is applied to both images
    #     # lambda x: transforms.RandomCrop(crop_size, fill=fillcolor)(x),
    #     # transforms.RandomAffine(degrees=0, translate=(0.05, 0), fill=fillcolor), # Shift height by 10%
    #     # transforms.RandomAffine(degrees=0, translate=(0, 0.05), fill=fillcolor),  # Shift width by 10%
    #     # RandomTransform(20, shift_range=10, fillcolor=0)
    #     transforms.RandomRotation(30)
    # ]
    scale_factor = random.uniform(0.7, 1.0)
    original_width, original_height = img.size
    crop_width = int(original_width * scale_factor)
    crop_height = int(original_height * scale_factor)
    crop_size = (crop_height, crop_width)
    
    mandatory_transform_list = [
        transforms.RandomCrop(size=(480, 480)), # Random cropping
    ]
    # Define transformations that will be applied to both input and seg images
    common_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Random affine transformation
        # transforms.RandomResizedCrop(size=(crop_size), scale=(0.7, 1.0)), # Random resizing and cropping
        # transforms.ElasticTransform(alpha=[50.0, 50.0], sigma=[5.0, 5.0]), # maybe remove
        transforms.ElasticTransform(alpha=[40.0, 40.0], sigma=[5.0, 5.0]), # maybe remove
        transforms.Lambda(lambda img: rotation_90(img)),
        transforms.Lambda(lambda img: rotation_180(img))
    ]
    
    img_transform_list = [
        transforms.Lambda(lambda img: brightness_adjustment(img, np.random.uniform(0.7, 1.3))), # Random brightness adjustment
        transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(np.random.uniform(0.8, 1.2))), # Random contrast adjustment (REMOVE IF BAD, TESTING)
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ]


    for i in range(100):    
        fillcolor = 0.
        random_seed = random.randint(0, 2**32 - 1)
        random.seed(random_seed)
        transformed_img = img.copy()
        transformed_seg = seg_img.copy()


        # Apply mandatory transformations to both images
        for transform in mandatory_transform_list:
            random_seed = random.randint(0, 2**32 - 1)
            if random.random() < 0.5:
                random.seed(random_seed)
                transformed_img = transform(transformed_img)
                random.seed(random_seed)
                transformed_seg = transform(transformed_seg)

        # Apply common transformations to both images
        for transform in common_transform_list:
            random_seed = random.randint(0, 2**32 - 1)
            if random.random() < 0.5:
                random.seed(random_seed)
                transformed_img = transform(transformed_img)
                random.seed(random_seed)
                transformed_seg = transform(transformed_seg)

        # Apply image-specific transformations only to the initial img
        for transform in img_transform_list:
            if random.random() < 0.5:
                transformed_img = transform(transformed_img)                                
        # Save the transformed images
        img_out_path = os.path.join(IMG_OUT_DIR, os.path.basename(img_path).replace('.', f'_{i}.'))
        seg_out_path = os.path.join(SEG_OUT_DIR, os.path.basename(seg_path).replace('.', f'_{i}.'))
        transformed_img.save(img_out_path)
        transformed_seg.save(seg_out_path)


def main():
    # Create output directories if they don't exist
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    os.makedirs(SEG_OUT_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_OUT_DIR, exist_ok=True)
    os.makedirs(VAL_SEG_OUT_DIR, exist_ok=True)

        
    # Clear existing images in the output directories
    clear_directory(IMG_OUT_DIR)
    clear_directory(SEG_OUT_DIR)
    clear_directory(VAL_IMG_OUT_DIR)
    
    for img_name, seg_name in zip(sorted(os.listdir(IMG_IN_DIR)), sorted(os.listdir(SEG_IN_DIR))):
        if (img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png") or img_name.endswith(".tif")) \
                and (seg_name.endswith(".jpg") or seg_name.endswith(".jpeg") or seg_name.endswith(".png") or seg_name.endswith(".gif")):
                img_path = os.path.join(IMG_IN_DIR, img_name)
                seg_path = os.path.join(SEG_IN_DIR, seg_name)
                generate_data(img_path, seg_path)

    # Iterate over each image in the validation directory
    for (img_name, seg_name) in zip(sorted(os.listdir(VAL_IMG_IN_DIR)), sorted(os.listdir(VAL_SEG_IN_DIR))):
        if (img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png") or img_name.endswith(".tif")) \
            and (seg_name.endswith(".jpg") or seg_name.endswith(".jpeg") or seg_name.endswith(".png") or seg_name.endswith(".gif")):
            img_path = os.path.join(VAL_IMG_IN_DIR, img_name)
            seg_path = os.path.join(VAL_SEG_IN_DIR, seg_name)
            img = cv2.imread(img_path)
            enhanced_img = enhance_veins_v4(img)
            img_out_path = os.path.join(VAL_IMG_OUT_DIR, os.path.basename(img_path))
            Image.fromarray(enhanced_img).save(img_out_path)


if __name__ == "__main__":
    main()
