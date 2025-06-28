import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.image_dir, img_id + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_id + "_segmentation.png")).convert("L")

        if self.transform:
            img = self.transform(img)
        mask = self.mask_transform(mask)

        return img, mask
