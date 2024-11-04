import torch
import torch.nn as nn

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        
        # Encoder con Batch Normalization
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Riduce la dimensione dell'immagine della metà
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Riduce la dimensione dell'immagine della metà
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Riduce la dimensione dell'immagine della metà
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Riduce la dimensione dell'immagine della metà
        )

        # Bottleneck Layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
        # Decoder con ConvTranspose e Skip Connections (utilizzando U-Net come riferimento)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=2, stride=2),  # Include le skip connections
            nn.ReLU()
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2),  # Include le skip connections
            nn.ReLU()
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2),  # Include le skip connections
            nn.ReLU()
        )
        
        # Output Layer per la DepthMap
        self.output_layer = nn.Conv2d(64 + 64, 1, kernel_size=1)  # Convoluzione per ottenere l'output con 1 canale
    
    def forward(self, x):
        # Encoder con salvataggio dei tensori per le skip connections
        enc1_out = self.enc1(x)  # Primo livello dell'encoder
        enc2_out = self.enc2(enc1_out)  # Secondo livello dell'encoder
        enc3_out = self.enc3(enc2_out)  # Terzo livello dell'encoder
        enc4_out = self.enc4(enc3_out)  # Quarto livello dell'encoder
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)

        # Decoder con Skip Connections
        dec4_out = self.dec4(bottleneck_out)

        # Utilizziamo l'interpolazione per assicurare che le dimensioni siano le stesse prima di concatenare
        if dec4_out.size() != enc4_out.size():
            enc4_out = torch.nn.functional.interpolate(enc4_out, size=dec4_out.shape[2:], mode='nearest')
        
        dec4_out = torch.cat((dec4_out, enc4_out), dim=1)  # Skip connection con output dell'encoder livello 4
        
        dec3_out = self.dec3(dec4_out)
        
        if dec3_out.size() != enc3_out.size():
            enc3_out = torch.nn.functional.interpolate(enc3_out, size=dec3_out.shape[2:], mode='nearest')
        
        dec3_out = torch.cat((dec3_out, enc3_out), dim=1)  # Skip connection con output dell'encoder livello 3
        
        dec2_out = self.dec2(dec3_out)

        if dec2_out.size() != enc2_out.size():
            enc2_out = torch.nn.functional.interpolate(enc2_out, size=dec2_out.shape[2:], mode='nearest')

        dec2_out = torch.cat((dec2_out, enc2_out), dim=1)  # Skip connection con output dell'encoder livello 2
        
        dec1_out = self.dec1(dec2_out)

        if dec1_out.size() != enc1_out.size():
            enc1_out = torch.nn.functional.interpolate(enc1_out, size=dec1_out.shape[2:], mode='nearest')

        dec1_out = torch.cat((dec1_out, enc1_out), dim=1)  # Skip connection con output dell'encoder livello 1
        
        # Output finale
        depth = self.output_layer(dec1_out)
        
        return depth

# Esempio di utilizzo
model = DepthEstimationModel()
