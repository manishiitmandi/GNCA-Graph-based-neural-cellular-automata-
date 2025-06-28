import torch.nn as nn
import torch
from gcn_lib.dense.torch_vertex import DynConv2d
# gcn_lib is downloaded from https://github.com/lightaime/deep_gcns_torch

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GrapherModule(nn.Module):
    """Grapher module with graph conv and FC layers
    """
    def __init__(self, in_channels, hidden_channels, k=9, dilation=1, drop_path=0.0):
        super(GrapherModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = nn.Sequential(
            DynConv2d(in_channels, hidden_channels, k, dilation, act=None),
            nn.BatchNorm2d(hidden_channels),
        nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 1).contiguous()
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x.reshape(B, C, H, W)
    
class FFNModule(nn.Module):
    """Feed-forward Network
    """
    def __init__(self, in_channels, hidden_channels, drop_path=0.0):
        super(FFNModule, self).__init__()
        self.fc1 = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 1, stride=1, padding=0),
        nn.BatchNorm2d(hidden_channels),
        nn.GELU()
        )
        self.fc2 = nn.Sequential(
        nn.Conv2d(hidden_channels, in_channels, 1, stride=1, padding=0),
        nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class ViGBlock(nn.Module):
    """ViG block with Grapher and FFN modules
    """
    def __init__(self, channels, k, dilation, drop_path=0.0):
        super(ViGBlock, self).__init__()
        self.grapher = GrapherModule(channels, channels * 2, k, dilation, drop_path)
        self.ffn = FFNModule(channels, channels * 4, drop_path)
    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
        return x
    

# !pip install numpy==1.24.3
# !pip install scipy
# !git clone https://github.com/lightaime/deep_gcns_torch.git
# !pip install -q torch torchvision
# # Clean up conflicting packages
# !pip uninstall -y torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
# # Install working CPU wheels compatible with Colab (PyTorch 2.0)
# pip install -q torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# pip install -q torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# pip install -q torch-geometric
# !source ./deep_gcns_torch/deepgcn_env_install.sh