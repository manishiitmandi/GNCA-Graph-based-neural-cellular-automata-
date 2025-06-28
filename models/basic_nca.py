import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicNCA(nn.Module):
    r"""Basic implementation of an NCA using a sobel x and y filter for the perception"""
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard"):
        super(BasicNCA, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        self.fc0 = nn.Linear(channel_n, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

    def perceive(self, x):
        r"""Perceptive function, combines 2 sobel x and y outputs with the identity of the cell
           Note: x is expected to be in format [B, H, W, C] after transposing in update
        """
        # Check channel dimension (last after transpose)
        #print(f"Basic perceive input shape: {x.shape}, expected channels: {self.channel_n}")
        
        def _perceive_with(x, weight):
            # x is [B, H, W, C]
            # Need to transpose to [B, C, H, W] for conv2d
            x_t = x.permute(0, 3, 1, 2)
            
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            
            # Apply convolution
            y = F.conv2d(x_t, conv_weights, padding=1, groups=self.channel_n)
            
            # Return to [B, H, W, C]
            return y.permute(0, 2, 3, 1)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T

        y1 = _perceive_with(x, dx)
        y2 = _perceive_with(x, dy)
        
        # Concatenate along channel dimension (last after transpose)
        y = torch.cat((x, y1, y2), dim=3)
        #print(f"Basic perceive output shape: {y.shape}")
        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image [B, C, H, W]
                fire_rate: random activation of cells
        """
        # Debug info
        #print(f"Update input shape: {x_in.shape}")
        
        # Transpose dimensions to [batch, height, width, channels]
        # This moves channels to the last dimension
        x = x_in.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        #print(f"After transpose in update: {x.shape}")
        
        # Apply perceive function - this returns [B, H, W, 3C]
        dx = self.perceive(x)
        #print(f"After perceive in update: {dx.shape}")
        
        # Apply fully connected layers
        # FC layers operate on the last dimension
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)
        #print(f"After FC layers: {dx.shape}")

        # Apply stochastic fire rate
        if fire_rate is None:
            fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1], device=self.device) > fire_rate
        stochastic = stochastic.float()
        dx = dx * stochastic

        # Update cells - add dx to x
        x = x + dx
        #print(f"After update (before transpose back): {x.shape}")
        
        # Return to original [B, C, H, W] format
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        #print(f"Update output shape (after transpose back): {x.shape}")
        
        return x

    def forward(self, x, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image [B, C, H, W]
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        # Debug info
        #print(f"Forward input shape: {x.shape}")
        
        # Ensure x has the right shape for our model [B, C, H, W]
        if x.shape[1] != self.channel_n:
            raise ValueError(f"Expected {self.channel_n} channels in BasicNCA.forward, got {x.shape[1]}")
        
        # Apply update steps
        for step in range(steps):
            # Apply update and clone to avoid in-place modification
            x2 = self.update(x, fire_rate).clone()
            
            # Preserve input channels if needed
            if self.input_channels > 0 and self.input_channels < self.channel_n:
                # Keep the original input channels from x and the updated rest from x2
                x = torch.cat((x[:, :self.input_channels], x2[:, self.input_channels:]), dim=1)
            else:
                x = x2
                
            # Debug info every few steps
            if step % 10 == 0:
                print(f"Step {step}, shape: {x.shape}")
                
        return x