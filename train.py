import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.mednca_highres import MedNCAHighRes
from datasets.isic_dataset import ISIC2018Dataset
from losses import DiceFocalLoss
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch.nn.functional as F
import faulthandler

# Create output directory for logs and checkpoints
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "output/training_{}".format(timestamp)
os.makedirs(output_dir, exist_ok=True)

# Set up device with better error handling
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Training configuration
config = {
    "batch_size": 2,
    "epochs": 10,
    "lr": 1e-3,
    "patch_size": 64,
    "image_size": 128,
    "channel_n": 16,
    "fire_rate": 0.5,
    "dice_weight": 0.7,
    "focal_weight": 0.3,
    "save_checkpoint_every": 5,  # Save model every 5 epochs
    "log_interval": 10,  # Log every 10 batches
}

transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor()
])

# Dataset and loader with error handling
try:
    train_dataset = ISIC2018Dataset(
        image_dir="data/ISIC2018_Task1-2_Training_Input",
        mask_dir="data/ISIC2018_Task1_Training_GroundTruth",
        transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=8,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    
    print(f"Loaded dataset with {len(train_dataset)} images")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Model initialization with error handling
try:
    model = MedNCAHighRes(
        channel_n=config["channel_n"], 
        fire_rate=config["fire_rate"], 
        device=device, 
        input_channels=1
    ).to(device)
    
    # Print model summary
    print(f"Model initialized: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
except Exception as e:
    print(f"Error initializing model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
criterion = DiceFocalLoss(dice_weight=config["dice_weight"], focal_weight=config["focal_weight"])

# Create a learning rate scheduler
# Create a learning rate scheduler (without verbose parameter for compatibility)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
print("Learning rate scheduler initialized")

# Initialize lists to store metrics for plotting
train_losses = []
epoch_losses = []

# Save config to file
with open(f"{output_dir}/config.txt", "w") as f:
    for key, value in config.items():
        f.write(f"{key}: {value}\n")

# Function to visualize predictions
def visualize_results(img, mask, pred, epoch, batch_idx, sample_idx):
    """Save visualization of input, ground truth and prediction"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img.cpu().permute(1, 2, 0).numpy())

    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred.cpu().detach().squeeze().numpy(), cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch{epoch+1}_batch{batch_idx}_sample{sample_idx}.png")
    plt.close()

# Training loop with progress bar and better error handling
print("Starting training...")
try:
    start_time = time.time()
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        batch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, (imgs, masks) in enumerate(progress_bar):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Forward pass (no patch coordinates needed for full mask prediction)
            try:
                preds = model(imgs, patch_coords=None)  # Should output (B, 1, 128, 128)

                # Resize ground truth masks to match prediction shape if needed
                if masks.shape != preds.shape:
                    masks = F.interpolate(masks, size=preds.shape[-2:], mode='bilinear', align_corners=False)
                
                print("Preds shape:", preds.shape)
                print("Masks shape:", masks.shape)

                loss = criterion(preds, masks)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Loss is NaN or Inf! Skipping batch {batch_idx}")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_losses.append(batch_loss)

                progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

                if batch_idx % config["log_interval"] == 0:
                    train_losses.append(batch_loss)

                    if batch_idx < 2:
                        sample_idx = 0
                        visualize_results(
                            imgs[sample_idx],
                            masks[sample_idx],
                            torch.sigmoid(preds[sample_idx]),
                            epoch,
                            batch_idx,
                            sample_idx
                        )

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} completed | Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if (epoch + 1) % config["save_checkpoint_every"] == 0:
            checkpoint_path = f"{output_dir}/model_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = f"{output_dir}/model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss (Batch)')
    plt.xlabel('Batch (x10)')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_losses)
    plt.title('Training Loss (Epoch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png")
    plt.close()
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")
    print(f"Final loss: {epoch_losses[-1]:.4f}")
    print(f"All outputs saved to {output_dir}")

except KeyboardInterrupt:
    print("Training interrupted by user")
    # Save interrupted model
    torch.save(model.state_dict(), f"{output_dir}/model_interrupted.pth")
    print(f"Model state saved to {output_dir}/model_interrupted.pth")

except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()

    faulthandler.enable()