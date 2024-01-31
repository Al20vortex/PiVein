import math
import os
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys
# from models.arm_model_v19 import CustomUNetArm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from torch.cuda.amp import autocast, GradScaler
from models.model_v54 import *
from torch.optim.lr_scheduler import CyclicLR
import csv
from torch.utils.tensorboard import SummaryWriter
from train_utils import *


CURRENT_MODE = MODE.V

HYPERPARAMS = {
    'IMAGE_DIMS': (512, 512),
    'INPUT_CHANNELS': 1,
    'BATCH_SIZE': 128,
    'EPOCHS': 100000,
    'PATIENCE': 10,
    'LEARNING_RATE': 1e-4,
    'OUTPUT_DIR': "output/main_v3",  # DUMMY VALUE, REPLACED IN MAIN
    'DROPOUT_RATE': 0.1,
    'WEIGHT_DECAY': 0.0,  # AKA L1/L2 Regularization Gamma
    'LOG_FILE_NAME': 'stats_training.csv'
}

device = torch.device("cuda")

transform = transforms.Compose([
    # transforms.Resize(HYPERPARAMS['IMAGE_DIMS']),
    transforms.CenterCrop(HYPERPARAMS['IMAGE_DIMS']),
    transforms.ToTensor(),
])

if __name__ == '__main__':    
    if CURRENT_MODE == MODE.V:
        HYPERPARAMS['OUTPUT_DIR'] = "output/veins_cldice"
        HYPERPARAMS['EPOCHS'] = 10000
        HYPERPARAMS['PATIENCE'] = 5
        HYPERPARAMS['BATCH_SIZE'] = 16
        HYPERPARAMS['LEARNING_RATE'] = 1e-4
        HYPERPARAMS['WEIGHT_DECAY'] = 1e-6
        seg_model = CustomUNet(HYPERPARAMS['INPUT_CHANNELS'], HYPERPARAMS['DROPOUT_RATE']).to(device)
        # arm_model = CustomUNetArm(1, HYPERPARAMS['DROPOUT_RATE']).to(device)
        
        # arm_model.load_state_dict(torch.load('output/arms/arm_model_v19/seg_model.pth'))
        # arm_model.eval()
        seg_optim = torch.optim.Adam(seg_model.parameters(), lr=HYPERPARAMS['LEARNING_RATE'])

        # train_targets_dir = '../seg_arm_veins_dataset/mask/train_aug_new'
        # train_img_dir = '../seg_arm_veins_dataset/img/train_aug_enhanced_new'
        # val_targets_dir = '../seg_arm_veins_dataset/mask/validation'
        # val_img_dir = '../seg_arm_veins_dataset/img/validation'
        
        # this is for the inhouse dataset
        train_img_dir = './datasets/vein_seg_dataset/images_aug'
        train_targets_dir = './datasets/vein_seg_dataset/targets_aug'
        val_img_dir = './datasets/vein_seg_dataset/val_images_aug'
        val_targets_dir = './datasets/vein_seg_dataset/val_targets'
        
        # this is for the VeinRL dataset
        # train_img_dir = './datasets/VeinRLDataset/images_aug'
        # train_targets_dir = './datasets/VeinRLDataset/targets_aug'
        # val_img_dir = './datasets/VeinRLDataset/val_images_aug'
        # val_targets_dir = './datasets/VeinRLDataset/val_targets'



    elif CURRENT_MODE == MODE.A:
        HYPERPARAMS['OUTPUT_DIR'] = "output/arms"
        HYPERPARAMS['EPOCHS'] = 10000
        HYPERPARAMS['LEARNING_RATE'] = 1e-3
        HYPERPARAMS['BATCH_SIZE'] = 196
        # seg_model = CustomUNetArm(HYPERPARAMS['INPUT_CHANNELS'], HYPERPARAMS['DROPOUT_RATE']).to(device)
        # seg_optim = torch.optim.Adam(seg_model.parameters(), lr=HYPERPARAMS['LEARNING_RATE'])
        arm_model = None
        train_targets_dir = './datasets/arm_seg_dataset/targets_aug'
        train_img_dir = './datasets/arm_seg_dataset/images_aug'
        val_targets_dir = './datasets/vein_seg_dataset/val_targets'
        val_img_dir = './datasets/vein_seg_dataset/val_images_aug'

    else:
        # CURRENT_MODE == MODE.D or O
        HYPERPARAMS['OUTPUT_DIR'] = "output/drive"
        HYPERPARAMS['EPOCHS'] = 10000
        HYPERPARAMS['BATCH_SIZE'] = 128
        seg_model = CustomUNet(HYPERPARAMS['INPUT_CHANNELS'], HYPERPARAMS['DROPOUT_RATE']).to(device)
        arm_model = None
        seg_optim = torch.optim.Adamax(seg_model.parameters(), lr=HYPERPARAMS['LEARNING_RATE'])
        train_targets_dir = './datasets/drive/training/mask_aug'
        train_img_dir = './datasets/drive/training/images_aug_enhanced'
        val_targets_dir = './datasets/drive/test/mask'
        val_img_dir = './datasets/drive/test/images'


    example_input = torch.rand(1, HYPERPARAMS['INPUT_CHANNELS'], HYPERPARAMS['IMAGE_DIMS'][0], HYPERPARAMS['IMAGE_DIMS'][1]).to(device)
    # summ_writer = SummaryWriter('runs/model_visualization')
    # summ_writer.add_graph(seg_model, example_input)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(seg_optim, mode='min', min_lr=1e-5, factor=0.75, patience=HYPERPARAMS['PATIENCE'], verbose=True)

    output_dir = HYPERPARAMS['OUTPUT_DIR'] + '/' + seg_model.name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_file_path = f'{output_dir}/{HYPERPARAMS["LOG_FILE_NAME"]}'

    # Get last completed epoch from CSV file
    last_completed_epoch = get_last_completed_epoch(csv_file_path)

    if last_completed_epoch == 0:
        # Create a CSV file and write the headers
        create_csv(csv_file_path)  

    # Build the dataset
    train_dataset = PairedImageDataset(train_img_dir, train_targets_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=HYPERPARAMS['BATCH_SIZE'], shuffle=True,
                                               pin_memory=True, drop_last=False)
    if CURRENT_MODE == MODE.V:
        val_dataset = PairedImageDataset(val_img_dir, val_targets_dir, transform=transform)
    else:
        val_dataset = PairedImageDatasetEnhanced(val_img_dir, val_targets_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=HYPERPARAMS['BATCH_SIZE'], pin_memory=True, drop_last=False, shuffle=True)
    scaler = GradScaler()  # scaler for AMP training
    
    # Check if pre-trained model exists
    if os.path.exists(f'{output_dir}/seg_model.pth'):
        # Load the pre-trained weights
        seg_model.load_state_dict(torch.load(f'{output_dir}/seg_model.pth'))
        print('Pre-trained model loaded successfully!')
        
        # Check if optimizer states exist and load them
        if os.path.exists(f'{output_dir}/seg_model_optim.pth'):
            seg_optim.load_state_dict(torch.load(f'{output_dir}/seg_model_optim.pth'))
            print('Optimizer state loaded successfully!')
        
        # Check if scaler states exist and load them
        if os.path.exists(f'{output_dir}/scaler.pth'):
            scaler.load_state_dict(torch.load(f'{output_dir}/scaler.pth'))
            print('Scaler state loaded successfully!')
    else:
        print('No pre-trained model found. Starting from scratch.')
    
    train(seg_model, seg_optim, scaler, scheduler, train_loader, val_loader, output_dir, last_completed_epoch, HYPERPARAMS, CURRENT_MODE, arm_model)
    
    
