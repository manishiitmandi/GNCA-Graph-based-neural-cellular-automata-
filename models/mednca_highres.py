import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_nca import BackboneNCA

class MedNCAHighRes(nn.Module):
    def __init__(self, channel_n=16, fire_rate=0.5, device="cuda", input_channels=3):
        super(MedNCAHighRes, self).__init__()
        self.device = device
        self.input_channels = input_channels
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.b1 = BackboneNCA(self.channel_n, fire_rate, device, input_channels=input_channels)
        self.b2 = BackboneNCA(self.channel_n, fire_rate, device, input_channels=input_channels + self.channel_n)

    def forward(self, img, patch_coords=None, steps_b1=48, steps_b2=48):
        # Step 1: Downsample image
        img_down = F.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        # Step 2: Ensure x has exactly `channel_n` channels
        if img_down.shape[1] < self.channel_n:
            pad_channels = self.channel_n - img_down.shape[1]
            x = torch.cat([img_down, torch.zeros(img_down.shape[0], pad_channels, *img_down.shape[2:], device=self.device)], dim=1)
        elif img_down.shape[1] > self.channel_n:
            x = img_down[:, :self.channel_n, :, :]
        else:
            x = img_down
        
        # Step 3: First NCA block (low resolution)
        x_b1 = x.clone()
        for i in range(steps_b1):
            x_b1 = self.b1.update(x_b1, self.fire_rate)  # Using the update function
        out_b1 = x_b1
        
        # Step 4: Upsample back to original resolution
        out_b1_up = F.interpolate(out_b1, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Step 5: Concatenate original image with upsampled channels (excluding original input channels)
        img_aug = torch.cat([img, out_b1_up[:, self.input_channels:]], dim=1)
        
        # Step 6: Ensure input to second NCA also has `channel_n` channels
        if img_aug.shape[1] > self.channel_n:
            patch_x = img_aug[:, :self.channel_n, :, :]
        elif img_aug.shape[1] < self.channel_n:
            pad = torch.zeros(img_aug.shape[0], self.channel_n - img_aug.shape[1],
                            img_aug.shape[2], img_aug.shape[3], device=self.device)
            patch_x = torch.cat([img_aug, pad], dim=1)
        else:
            patch_x = img_aug
        
        # Step 7: Second NCA block (full resolution)
        x_b2 = patch_x.clone()
        for i in range(steps_b2):
            x_b2 = self.b2.update(x_b2, self.fire_rate)  # Using the update function
        out_b2 = x_b2
        
        """# Print shapes for debugging
        print(f"Input shape: {img.shape}")
        print(f"Downsampled shape: {img_down.shape}")
        print(f"B1 input shape: {x.shape}")
        print(f"B1 output shape: {out_b1.shape}")
        print(f"B1 upsampled shape: {out_b1_up.shape}")
        print(f"Augmented image shape: {img_aug.shape}")
        print(f"B2 input shape: {patch_x.shape}")
        print(f"B2 output shape: {out_b2.shape}")"""
        
        return out_b2[:, :1]  # Output single channel mask