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
            # First ConvTranspose2d Layer: (32, 512, 7, 7) -> (32, 256, 14, 14)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second ConvTranspose2d Layer: (32, 256, 14, 14) -> (32, 128, 28, 28)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            
            # Third ConvTranspose2d Layer: (32, 128, 28, 28) -> (32, 64, 56, 56)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Fourth ConvTranspose2d Layer: (32, 64, 56, 56) -> (32, 32, 112, 112)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=1, stride=1),
            
            # Fifth ConvTranspose2d Layer for final upsampling: (32, 32, 112, 112) -> (32, 1, 224, 224)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(1, 1, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        # Encoder pass
        features = self.encoder_layers(x)  # (batch_size, 512, 7, 7)
        
        # Decoder pass
        depth = self.decoder(features)     # Output finale atteso: (batch_size, 1, 224, 224)  
        
        return depth
