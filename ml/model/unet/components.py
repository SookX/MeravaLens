import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    """
    A block that applies two consecutive convolutional layers, each followed by 
    Batch Normalization and ReLU activation.

    This is the basic building block used in U-Net encoder and decoder paths.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after convolution.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Padding added to all sides of the input. Default is 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class FeedForwardBlock(nn.Module):
    """
    A fully connected feedforward classifier that flattens the spatial feature map 
    and passes it through a two-layer MLP.

    Typically used after the bottleneck in U-Net for high-level classification (e.g., domain type).

    Args:
        in_channels (int): Number of channels in the feature map.
        height (int, optional): Height of the input feature map. Default is 64.
        width (int, optional): Width of the input feature map. Default is 64.
    """
    def __init__(self, in_channels, height=32, width=32):
        super().__init__()
        self.flatten_dim = in_channels * height * width
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1),
        )
    
    def forward(self, x):
        return self.fc(x)
    
class TransposeConvBlock(nn.Module):
    """
    A decoder block that performs upsampling using a transposed convolution, 
    concatenates with a skip connection, and applies a DoubleConvBlock.

    Used in the expansive path of U-Net to recover spatial resolution and refine features.

    Args:
        in_channels (int): Number of input channels to the transposed convolution.
        out_channels (int): Number of output channels after upsampling and refinement.
        kernel_size (int, optional): Kernel size for the transposed convolution. Default is 2.
        stride (int, optional): Stride for the transposed convolution. Default is 2.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 2, stride = 2):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride)
        self.double_conv = DoubleConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip_connection):
        x = self.transposed_conv(x)
        x  = torch.cat((skip_connection, x), dim=1)
        x = self.double_conv(x)
        return x

if __name__ == "__main__":
    in_channels = 64
    out_channels = 32
    x = torch.randn(1, in_channels, 32, 32)           
    skip = torch.randn(1, in_channels, 64, 64)        

    block = TransposeConvBlock(in_channels, out_channels)
    output = block(x, skip)

    

    print("Output shape:", output.shape) 