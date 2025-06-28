import torch
from train import model, device, transform  # reuse model and transform
from utils import ISICDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets.isic_dataset import ISIC2018Dataset

test_dataset = ISIC2018Dataset(
    image_dir="data/ISIC2018_Task1-2_Test_Input",
    mask_dir="data/ISIC2018_Task1_Test_GroundTruth",
    transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def show_segmentation(img, pred_mask, true_mask):
    img = img.cpu().squeeze().permute(1, 2, 0).numpy()
    pred = pred_mask.cpu().squeeze().numpy()
    true = true_mask.cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title("Input Image")
    axs[1].imshow(pred, cmap="gray")
    axs[1].set_title("Predicted")
    axs[2].imshow(true, cmap="gray")
    axs[2].set_title("Ground Truth")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

model.eval()
with torch.no_grad():
    for i, (imgs, masks) in enumerate(test_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        x = torch.cat([imgs, torch.zeros(imgs.shape[0], 16 - imgs.shape[1], *imgs.shape[2:])], dim=1)
        x = x.permute(0, 2, 3, 1)

        out = model(x, steps=64, fire_rate=0.5)
        pred = torch.sigmoid(out[..., 0])

        show_segmentation(imgs[0], pred[0], masks[0])
        if i == 4:
            break  # Show only 5 samples
