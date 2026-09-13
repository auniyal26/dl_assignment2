import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskPetCNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        # Shared encoder
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Classification branch
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

        # Segmentation decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(128 + 256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(64 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(32 + 64, 32)

        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32)

        self.seg_out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                 # 32 x 128 x 128
        e2 = self.enc2(self.pool(e1))     # 64 x 64 x 64
        e3 = self.enc3(self.pool(e2))     # 128 x 32 x 32
        e4 = self.enc4(self.pool(e3))     # 256 x 16 x 16
        bottleneck = self.pool(e4)        # 256 x 8 x 8

        cls_logits = self.classifier(self.cls_pool(bottleneck))

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg_logits = self.seg_out(d1)

        return cls_logits, seg_logits
