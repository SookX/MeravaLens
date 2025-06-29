import torch
import torch
import torch.nn as nn
from torchvision.models import resnet18
from .components import TransposeConvBlock


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


if __name__ == "__main__":
    batch_size = 8
    in_channels = 3
    out_channels = 7
    height, width = 512, 512

    model = ResUNet(in_channels=in_channels, out_channels=out_channels).to("cuda")

    x = torch.randn(batch_size, in_channels, height, width).to("cuda")

    
    torch.cuda.empty_cache()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    y = model(x)