import torch
import torch.nn as nn
from torchvision.models import resnet18

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
        self.double_conv = None
        if(in_channels == out_channels):
            self.double_conv = DoubleConvBlock(in_channels=in_channels * 2, out_channels=out_channels)
        else:
            self.double_conv = DoubleConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip_connection):
        x = self.transposed_conv(x)
        x  = torch.cat((skip_connection, x), dim=1)
        x = self.double_conv(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 7, initial_feature = 32, steps = 4):
        super().__init__()
        resnet = resnet18(pretrained=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  
            nn.Sequential(resnet.maxpool, resnet.layer1),          
            resnet.layer2,                                        
            resnet.layer3,                                         
            resnet.layer4                                          
        ])

        self.fc = nn.Sequential(
            resnet.avgpool,    
            nn.Flatten(),
            nn.Linear(512, 1)
        )

        self.decoder = nn.ModuleList([
            TransposeConvBlock(512, 256),  
            TransposeConvBlock(256, 128),  
            TransposeConvBlock(128, 64),   
            TransposeConvBlock(64, 64),   
        ])

        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        bottleneck = skip_connections.pop() 

        label = self.fc(bottleneck)
        x = bottleneck
        skip_connections = skip_connections[::-1]

        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[i]
            x = decoder_block(x, skip)

        x = self.final_upsample(x)  
        output = self.out_conv(x)

        return label, output


    def forward(self, x):
        skip_connections = []

        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        bottleneck = skip_connections.pop()

        label = self.fc(bottleneck) 
        x = bottleneck
        skip_connections = skip_connections[::-1]

        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[i]
            x = decoder_block(x, skip)

        x = self.final_upsample(x)
        output = self.out_conv(x)
        return label, output
