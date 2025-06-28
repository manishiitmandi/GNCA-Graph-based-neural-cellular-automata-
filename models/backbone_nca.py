import torch
import torch.nn as nn
from models.basic_nca import BasicNCA
#from models.vig_block import ViGBlock

class BackboneNCA(BasicNCA):
    r"""Implementation of the backbone NCA of Med-NCA"""
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, kernel_size=3):
        super(BackboneNCA, self).__init__(channel_n, fire_rate, device, hidden_size, input_channels)
        
        self.device = device
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        
        # Define convolutional perception layers
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, padding_mode="reflect")
        #self.vig = ViGBlock(channels=channel_n, k=9, dilation=1, drop_path=0.0)
        
        # Reduce combined channels to original channel count
        self.perceive_reduce = nn.Conv2d(4 * channel_n, channel_n, kernel_size=1)

        # Fully connected layers
        self.fc0 = nn.Linear(channel_n, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()
        
        # Move to device
        self.to(device)

    def perceive(self, x):
        # x: [B, H, W, C] -> convert to [B, C, H, W]
        x_conv = x.permute(0, 3, 1, 2)

        # Apply convolution layers
        y1 = self.p0(x_conv)
        y2 = self.p1(x_conv)
        #y_vig = self.vig(x_conv)

        # Concatenate: [B, 4*C, H, W]
        #y = torch.cat([x_conv, y1, y2, y_vig], dim=1)
        y = torch.cat([x_conv, y1, y2], dim=1)

        # Reduce to original channel size
        y = self.perceive_reduce(y)

        # Convert back to [B, H, W, C]
        return y.permute(0, 2, 3, 1)

    def update(self, x_in, fire_rate=None):
        # x_in: [B, C, H, W] -> [B, H, W, C]
        x = x_in.permute(0, 2, 3, 1)

        # Perceive neighbors
        y = self.perceive(x)  # [B, H, W, C]

        # Flatten spatial dims to apply Linear
        # Flatten spatial dims to apply Linear
        B, H, W, C = y.shape
        y_flat = y.reshape(-1, C)  # Safe reshape for non-contiguous input


        # Apply MLP
        dx = self.fc0(y_flat)
        dx = torch.relu(dx)
        dx = self.fc1(dx)

        # Reshape back to [B, H, W, C]
        dx = dx.view(B, H, W, C)

        # Apply stochastic fire rate
        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = (torch.rand(B, H, W, 1, device=self.device) <= fire_rate).float()
        dx = dx * stochastic

        # Update state
        x = x + dx

        # Return to [B, C, H, W]
        return x.permute(0, 3, 1, 2)
