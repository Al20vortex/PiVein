import math
import os
from sklearn.metrics import roc_auc_score
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from torch.cuda.amp import autocast, GradScaler
import csv
from enum import Enum, auto
from cldice_loss.pytorch.cldice import soft_dice_cldice
class MODE(Enum):
    V = auto()
    A = auto()
    D = auto()
    R = auto()
    O = auto()


device = torch.device('cuda')

def train(seg_model, seg_optim, scaler, scheduler, train_loader, val_loader, output_dir, last_completed_epoch, HYPERPARAMS, mode=MODE.A, arm_model=None):
    
    IMAGE_DIMS = HYPERPARAMS['IMAGE_DIMS']
    INPUT_CHANNELS = HYPERPARAMS['INPUT_CHANNELS']
    BATCH_SIZE = HYPERPARAMS['BATCH_SIZE']
    EPOCHS = HYPERPARAMS['EPOCHS']
    PATIENCE = HYPERPARAMS['PATIENCE']
    LEARNING_RATE = HYPERPARAMS['LEARNING_RATE']
    OUTPUT_DIR = HYPERPARAMS['OUTPUT_DIR']
    DROPOUT_RATE = HYPERPARAMS['DROPOUT_RATE']
    WEIGHT_DECAY = HYPERPARAMS['WEIGHT_DECAY']
    LOG_FILE_NAME = HYPERPARAMS['LOG_FILE_NAME']


    best_loss = float('inf')
    val_loss = float('inf')
    epsilon = 1e-7  # small constant to avoid division by zeros
    dice_cldice_loss = soft_dice_cldice()  # initialize loss


    for epoch in range(last_completed_epoch, EPOCHS):
        # Initialize metrics variables for training
        train_true_pos, train_false_pos, train_false_neg, train_true_neg = 0, 0, 0, 0
        train_dice_scores = []  # Initialize list to store dice scores for training
        # Initialize metrics variables for validation
        val_true_pos, val_false_pos, val_false_neg, val_true_neg = 0, 0, 0, 0
        val_dice_scores = []  # Initialize list to store dice scores for validation
        
        
        start_time = time.time()
        # Training
        for (img, target) in train_loader:
            img = img.to(device)
            target = target.to(device)
            if mode == MODE.A or mode == MODE.V or mode == MODE.D:
                img = add_gaussian_noise(img, True, 0, 0.02).to(device)  # add gaussian noise to every image to improve generalization
            # if mode == MODE.V:
            #     arm_mask = torch.sigmoid(arm_model(img))
            #     img = img*arm_mask
            #     target = target*arm_mask

            for i in range(img.shape[0]):
                # if mode == MODE.D:
                #     for _ in range(random.randint(0, 1)):  # add a random amount of black boxes to the image
                #         img[i] = add_black_box(img[i], (64,64))
                #         img[i] = add_shade_box(img[i], (128,128))
                if mode == MODE.A:
                    for _ in range(random.randint(0, 1)):  # add a random amount of black boxes to the image
                        img[i] = add_black_box(img[i])
                        img[i] = add_shade_box(img[i])

                if mode == MODE.V:
                    rotate = RandomRotationTransform(10, fillcolor=0)
                    pass
                elif mode == MODE.A:
                    rotate = RandomRotationTransform(360, fillcolor=0)
                else:
                    rotate = RandomRotationTransform(4, fillcolor=0)
                # # Apply rotation
                img[i], target[i] = rotate(img[i], target[i])
                target[i] = (target[i]>0.5).float().to(device)
            """Calculate Training Loss"""
            seg_model.zero_grad()  
            
            with autocast(enabled=True):
                predicted_veins = seg_model(img)
                train_dice_loss = dice_cldice_loss(target, torch.sigmoid(predicted_veins))
                if mode == MODE.O:
                    train_loss = train_dice_loss + 0.5*nn.BCEWithLogitsLoss()(predicted_veins, target) + WEIGHT_DECAY*calculate_l1_norm(seg_model)
                else:
                    train_loss = train_dice_loss + WEIGHT_DECAY*calculate_l1_norm(seg_model)

            scaler.scale(train_loss).backward()
            scaler.step(seg_optim)
            scaler.update()
            
            # Check for NaNs in model parameters
            for name, param in seg_model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in {name}")
                    param.data.fill_(0)  # setting NaN values to zero
            
            # Update metrics variables for training
            train_dice_score = 1. - train_dice_loss.item()
            train_dice_scores.append(train_dice_score)  # Store dice score for each training batch            
            pred_binary = (torch.sigmoid(predicted_veins) > 0.5).int()
            target_binary = (target > 0.5).int()
            train_true_pos += torch.sum(pred_binary * target_binary).item()
            train_false_pos += torch.sum(pred_binary * (1 - target_binary)).item()
            train_false_neg += torch.sum((1 - pred_binary) * target_binary).item()
            train_true_neg += torch.sum((1 - pred_binary) * (1 - target_binary)).item()        
        
        # Validation
        for (val_img, val_target) in val_loader:     
            val_img = val_img.to(device)
            val_target = val_target.to(device)
            # if mode == MODE.V:
            #     arm_mask = torch.sigmoid(arm_model(val_img))
            #     val_img = val_img*arm_mask
            #     val_target = val_target*arm_mask

            """Calculate Validation Loss"""
            with torch.no_grad():
                seg_model.eval()
                
                with autocast(enabled=True):
                    val_preds = seg_model(val_img)
                    dice_loss = dice_cldice_loss(val_target, torch.sigmoid(val_preds))
                    if mode == MODE.O:
                        val_loss = dice_loss + 0.5*nn.BCEWithLogitsLoss()(val_preds, val_target) + WEIGHT_DECAY*calculate_l1_norm(seg_model)
                    else:
                        val_loss = dice_loss + WEIGHT_DECAY*calculate_l1_norm(seg_model)
                                
                seg_model.train()
            # Update metrics variables for validation
            val_dice_score = 1. - dice_loss.item()
            val_dice_scores.append(val_dice_score)  # Store dice score for each validation batch
            val_pred_binary = (torch.sigmoid(val_preds) > 0.5).int()
            val_target_binary = (val_target > 0.5).int()
            val_true_pos += torch.sum(val_pred_binary * val_target_binary).item()
            val_false_pos += torch.sum(val_pred_binary * (1 - val_target_binary)).item()
            val_false_neg += torch.sum((1 - val_pred_binary) * val_target_binary).item()
            val_true_neg += torch.sum((1 - val_pred_binary) * (1 - val_target_binary)).item()
        scheduler.step(val_loss)        
        
        # Calculate metrics for training
        train_precision = (train_true_pos + epsilon) / (train_true_pos + train_false_pos + epsilon)
        train_recall = (train_true_pos + epsilon) / (train_true_pos + train_false_neg + epsilon)
        train_accuracy = (train_true_pos + train_true_neg + epsilon) / (train_true_pos + train_true_neg + train_false_pos + train_false_neg + epsilon)
        train_auc = roc_auc_score(target_binary.cpu().numpy().flatten(), torch.sigmoid(predicted_veins).cpu().detach().numpy().flatten())
        train_mIoU = (train_true_pos + epsilon) / (train_true_pos + train_false_pos + train_false_neg + epsilon)

        # Calculate metrics for validation
        val_precision = (val_true_pos + epsilon) / (val_true_pos + val_false_pos + epsilon)
        val_recall = (val_true_pos + epsilon) / (val_true_pos + val_false_neg + epsilon)
        val_accuracy = (val_true_pos + val_true_neg + epsilon) / (val_true_pos + val_true_neg + val_false_pos + val_false_neg + epsilon)
        val_auc = roc_auc_score(val_target_binary.cpu().numpy().flatten(), torch.sigmoid(val_preds).cpu().detach().numpy().flatten())
        val_mIoU = (val_true_pos + epsilon) / (val_true_pos + val_false_pos + val_false_neg + epsilon)        
        # Calculate average dice score for the entire training and validation epochs
        avg_train_dice_score = sum(train_dice_scores) / len(train_dice_scores)
        avg_val_dice_score = sum(val_dice_scores) / len(val_dice_scores)
        
        if epoch % 1 == 0:
            # Open the CSV file in append mode and write the training stats every x epochs and save images
            with torch.no_grad():
                val_preds_expanded = val_preds.expand(-1, INPUT_CHANNELS, -1, -1)
                predicted_veins_expanded = predicted_veins.expand(-1, INPUT_CHANNELS, -1, -1)
                # save_sigmoid_image(torch.cat([mini_out, img], 0), f'{output_dir}/miniout_{epoch:04}.png')
                save_nonsigmoid_image(torch.cat([predicted_veins_expanded, img], 0), f'{output_dir}/train_{epoch:04}.png')
                save_nonsigmoid_image(torch.cat([val_preds_expanded, val_img], 0), f'{output_dir}/val_{epoch:04}.png')  # concat the original images for viewing
        
        
        # Check if validation loss improved
        if val_loss and val_loss < best_loss:
            best_val_dsc = val_dice_score
            if epoch > 1 and epoch % 1 == 0:
                best_loss = val_loss
                print("Saving best model")
                # save model states when we beat a previous record
                torch.save(seg_model.state_dict(), f'{output_dir}/seg_model.pth')
                # Save optimizer states
                torch.save(seg_optim.state_dict(), f'{output_dir}/seg_model_optim.pth')
                # Save scaler state
                torch.save(scaler.state_dict(), f'{output_dir}/scaler.pth')
        
        end_time = time.time()
        # print the losses and epoch time at the end of each epoch
        epoch_time = end_time - start_time

        with open(f'{output_dir}/{LOG_FILE_NAME}', "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss.item(), val_loss.item(), train_accuracy, val_accuracy, best_val_dsc, f'{seg_optim.param_groups[0]["lr"]:.2e}', seg_model.name, f'{WEIGHT_DECAY:.2e}', train_precision, train_recall, train_auc, train_mIoU, val_precision, val_recall, val_auc, val_mIoU, avg_train_dice_score, avg_val_dice_score, f'{epoch_time:.2f}s'])        
        print(
            f'[{epoch}/{EPOCHS}]  '
            f'Train_Loss: {train_loss.item():.4f}   '
            f'Val_Loss: {val_loss.item():.4f}   '
            f'Train_Accuracy: {train_accuracy:.4f}   '
            f'Val_Accuracy: {val_accuracy:.4f}   '
            # f'Train_Precision: {train_precision:.4f}   '
            # f'Val_Precision: {val_precision:.4f}   '
            # f'Train_Recall: {train_recall:.4f}   '
            # f'Val_Recall: {val_recall:.4f}   '
            # f'Train_AUC: {train_auc:.4f}   '
            # f'Val_AUC: {val_auc:.4f}   '
            f'Train_mIoU: {train_mIoU:.4f}   '
            f'Val_mIoU: {val_mIoU:.4f}   '
            f'Train_Dice_Score: {avg_train_dice_score:.4f}   '
            f'Val_Dice_Score: {avg_val_dice_score:.4f}   '
            f'Epoch Time: {epoch_time:.2f}s')