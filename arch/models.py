import torch
from torch import nn

class Autoencoder(nn.Module): 
    def __init__(self,up_layer='transpose', features=32):
        super(Autoencoder,self).__init__()
        self.encoder1 = UnetBlock(1,            features * 1)
        self.encoder2 = UnetBlock(features * 1, features * 2)
        self.encoder3 = UnetBlock(features * 2, features * 4)
        self.encoder4 = UnetBlock(features * 4, features * 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UnetBlock(features * 8, features * 16)

        if up_layer == 'shuffle':    
          self.upconv4 = nn.Sequential(
              nn.Conv2d(features * 16, features * 16 * 2, kernel_size=1, stride=1),
              nn.PixelShuffle(2) )
          self.upconv3 = nn.Sequential(
              nn.Conv2d(features * 8, features * 8 * 2, kernel_size=1, stride=1),
              nn.PixelShuffle(2) )
          self.upconv2 = nn.Sequential(
              nn.Conv2d(features * 4, features * 4 * 2, kernel_size=1, stride=1),
              nn.PixelShuffle(2) )
          self.upconv1 = nn.Sequential(
              nn.Conv2d(features * 2, features * 2 * 2, kernel_size=1, stride=1),
              nn.PixelShuffle(2) )
        elif up_layer == 'transpose':
          self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
          self.upconv3 = nn.ConvTranspose2d(features * 8,  features * 4, kernel_size=2, stride=2)
          self.upconv2 = nn.ConvTranspose2d(features * 4,  features * 2, kernel_size=2, stride=2)
          self.upconv1 = nn.ConvTranspose2d(features * 2,  features * 1, kernel_size=2, stride=2)

        self.decoder4 = UnetBlock((features * 8) * 2 , features * 8)
        self.decoder3 = UnetBlock((features * 4) * 2 , features * 4)
        self.decoder2 = UnetBlock((features * 2) * 2 , features * 2)
        self.decoder1 = UnetBlock((features * 1) * 2 , features * 1)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=1, kernel_size=1
        )


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        #dec4 = dec4 + enc4
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        #dec3 = dec3 + enc3
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        #dec2 = dec2 + enc2
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        #dec1 = dec1 + enc1
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

class UnetBlock(nn.Module):
    def __init__(self, channels, features):
        super(UnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu1(residual)
        #residual = self.conv2(residual)
        #residual = self.bn2(residual)
        #residual = self.prelu2(residual)

        return residual

class ResidualNet(nn.Module):
    def __init__(self,):

        super(ResidualNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU() 
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block8 = nn.Conv2d(64, 1, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2  #Output in [0, 1] range

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual