import os

import torch
import torch.nn as nn

from .components import DoubleConvBlock, FeedForwardBlock, TransposeConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 7, initial_feature = 32, steps = 4):
        """
        U-Net model for semantic segmentation with an auxiliary feedforward classifier.

        This implementation includes:
        - A contracting (encoder) path with repeated DoubleConvBlocks and max pooling.
        - A bottleneck DoubleConvBlock.
        - A FeedForwardBlock classifier to predict high-level labels (e.g., 'Rural' or 'Urban').
        - An expansive (decoder) path using TransposeConvBlocks and skip connections.
        - A final 1x1 convolution to map features to output segmentation classes.

        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB).
            out_channels (int): Number of output segmentation classes.
            initial_feature (int): Number of filters in the first encoder block.
            steps (int): Number of downsampling/upsampling steps (depth of the network).

        Returns:
            Tuple[Tensor, Tensor]: 
                - label: Output from the FeedForwardBlock of shape [batch_size, 1]
                - output: Segmentation map of shape [batch_size, out_channels, H, W]
        """
        super().__init__()
        self.contracting = nn.ModuleList()
        self.expansive = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        in_feat = in_channels
        out_feat = initial_feature

        for _ in range(steps):
            self.contracting.append(DoubleConvBlock(in_feat, out_feat))
            in_feat = out_feat
            out_feat = out_feat * 2

        self.bottleneck = DoubleConvBlock(in_feat, out_feat)
        self.ff = FeedForwardBlock(out_feat)

        for _ in range(steps):
            self.expansive.append(TransposeConvBlock(out_feat, in_feat))
            out_feat = in_feat
            in_feat = in_feat // 2

        self.out_conv = nn.Conv2d(in_channels=out_feat, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            Tuple[Tensor, Tensor]: 
                - label: FeedForwardBlock prediction of shape [batch_size, 1].
                - output: Segmentation map of shape [batch_size, out_channels, height, width].
        """
        skip_connections = []
        for layer in self.contracting:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        label = self.ff(x) # Detects either Rural or Urban
        skip_connections = list(reversed(skip_connections))
        for i, layer in enumerate(self.expansive):
            x = layer(x, skip_connections[i])

        
        output = self.out_conv(x)
        return label, output

        

if __name__ == "__main__":
    batch_size = 8
    in_channels = 3
    out_channels = 7
    height, width = 512, 512

    model = UNet(in_channels=in_channels, out_channels=out_channels).to("cuda")

    x = torch.randn(batch_size, in_channels, height, width).to("cuda")

    
    torch.cuda.empty_cache()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    y = model(x)

