import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from models.mednca_highres import MedNCAHighRes
from datasets.isic_dataset import ISIC2018Dataset
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch.nn.functional as F

# CONFIG
checkpoint_path = "/home/teaching/mednca_scratch_arnavk/output/training_20250520_142053/model_interrupted.pth"  # <-- Update this path
output_dir = f"output/testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# DATASET
test_dataset = ISIC2018Dataset(
    image_dir="/scratch/arnavk.scee.iitmandi/dataset_new/ISIC2018_Task1-2_Test_Input",
    mask_dir="/scratch/arnavk.scee.iitmandi/dataset_new/ISIC2018_Task1_Test_GroundTruth",
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# MODEL
# model = MedNCAHighRes(channel_n=16, fire_rate=0.5, device=device, input_channels=1)
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.to(device)
# model.eval()
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
model = MedNCAHighRes(
    channel_n=config["channel_n"], 
    fire_rate=config["fire_rate"], 
    device=device, 
    input_channels=1
).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# VISUALIZATION
def visualize(img, mask, pred, idx, dice_score):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(img.cpu().permute(1, 2, 0).numpy())

    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Prediction\nDice: {dice_score:.4f}")
    plt.imshow(pred.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_{idx}.png")
    plt.close()

# DICE SCORE FUNCTION
def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

# INFERENCE LOOP
print("Starting inference...")
all_dice_scores = []

with torch.no_grad():
    for idx, (img, mask) in enumerate(test_loader):
        img = img.to(device)  # (1, 1, 128, 128)
        mask = mask.to(device)  # (1, 1, 128, 128)

        # Create dummy patch coordinates (not used now, but model expects it)
        patch_coords = torch.tensor([[0, 0]], device=device)

        # Model prediction: Pass full image instead of patch
        pred_full = model(img, patch_coords)  # Output shape: (1, H, W)
        pred_sigmoid = torch.sigmoid(pred_full).unsqueeze(1)  # shape: (1, 1, H, W)

        # Compute Dice score on full image
        score = dice_score(pred_sigmoid, mask)
        all_dice_scores.append(score)

        # Visualize full prediction
        visualize(img[0], mask[0, 0], pred_sigmoid[0, 0], idx, score)

        # Save prediction
        np.save(f"{output_dir}/pred_mask_{idx}.npy", pred_sigmoid[0, 0].cpu().numpy())

# Summary
avg_dice = np.mean(all_dice_scores)
print(f"Inference completed on {len(test_dataset)} samples.")
print(f"Average Dice Score: {avg_dice:.4f}")
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"Average Dice Score: {avg_dice:.4f}\n")

