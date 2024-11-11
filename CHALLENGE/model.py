import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        
        # Utilizza ResNet34 pre-addestrato come encoder
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.encoder_layers = nn.Sequential(*list(resnet.children())[:-2])  # Remove the fully connected layer

        freeze_until = 4

        # Congelare i primi 'freeze_until' layer
        layer_count = 0
        for child in resnet.children():
            layer_count += 1
            if layer_count <= freeze_until:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Decoder per upsample
        self.decoder = nn.Sequential(
            # First ConvTranspose2d Layer: (32, 512, 5, 8) -> (32, 256, 10, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second ConvTranspose2d Layer: (32, 256, 10, 16) -> (32, 128, 20, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third ConvTranspose2d Layer: (32, 128, 20, 32) -> (32, 64, 40, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
  
            # Fourth ConvTranspose2d Layer: (32, 64, 40, 64) -> (32, 1, 74, 128)
            nn.ConvTranspose2d(64, 1, kernel_size=(2,3), stride=(2,2), padding=(3,1), output_padding=(0,1)),
            
            # Fifth ConvTranspose2d Layer for final upsampling: (32, 1, 74, 128) -> (32, 1, 144, 256)
            nn.ConvTranspose2d(1, 1, kernel_size=(2,3), stride=(2,2), padding=(2,1), output_padding=(0,1))
        )
    
    def forward(self, x):
        # Encoder pass
        features = self.encoder_layers(x)  # (32, 512, 5, 8)
        
        # Decoder pass
        depth = self.decoder(features)     # Output finale atteso: (32, 1, 144, 256)
        
        return depth
