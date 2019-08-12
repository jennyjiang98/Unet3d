# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

        self.up1r = up_reconstruct(512, 256)
        self.up2r = up_reconstruct(256, 128)
        self.up3r = up_reconstruct(128, 64)
        self.up4r = up_reconstruct(64, 32)
        self.outr = outconv(32, n_channels)
        self.outr2 = outconv(64, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)


        y = self.up1r(x5)
        y = self.up2r(y)
        y3 = self.up3r(y)
        y2 = self.up4r(y3)
        y1 = self.outr(y2)
        y = self.outr2(y3)
        ###decoder??

        return F.sigmoid(x), F.sigmoid(y1),F.sigmoid(y)
