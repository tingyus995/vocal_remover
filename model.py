import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2, padding: int = 2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.main(x)

class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2, padding: int = 2, output_padding: int = 1, act: bool = True, bn: bool = True):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU() if act else nn.Identity()
        )
    
    def forward(self, x):
        return self.main(x)

class Net(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.initial_bn = nn.BatchNorm2d(1)

        self.enc_1 = EncoderLayer(1, 16, 5, 2, 2)
        self.enc_2 = EncoderLayer(16, 32, 5, 2, 2)
        self.enc_3 = EncoderLayer(32, 64, 5, 2, 2)
        self.enc_4 = EncoderLayer(64, 128, 5, 2, 2)
        self.enc_5 = EncoderLayer(128, 256, 5, 2, 2)
        self.enc_6 = EncoderLayer(256, 512, 5, 2, 2)

        self.dec_1 = DecoderLayer(512, 256, 5, 2, 2)
        self.dec_2 = DecoderLayer(2 * 256, 128, 5, 2, 2)
        self.dec_3 = DecoderLayer(2 * 128, 64, 5, 2, 2)
        self.dec_4 = DecoderLayer(2 * 64, 32, 5, 2, 2)
        self.dec_5 = DecoderLayer(2 * 32, 16, 5, 2, 2)
        self.dec_6 = DecoderLayer(2 * 16, 1, 5, 2, 2, act = False, bn = False)

    def forward(self, input):
        
        x = self.initial_bn(input)
        enc_1_out = self.enc_1(x)
        enc_2_out = self.enc_2(enc_1_out)
        enc_3_out = self.enc_3(enc_2_out)
        enc_4_out = self.enc_4(enc_3_out)
        enc_5_out = self.enc_5(enc_4_out)
        enc_6_out = self.enc_6(enc_5_out)

        x = self.dec_1(enc_6_out)
        x = torch.cat([enc_5_out, x], dim=1)
        x = self.dec_2(x)
        x = torch.cat([enc_4_out, x], dim=1)
        x = self.dec_3(x)
        x = torch.cat([enc_3_out, x], dim=1)
        x = self.dec_4(x)
        x = torch.cat([enc_2_out, x], dim=1)
        x = self.dec_5(x)
        x = torch.cat([enc_1_out, x], dim=1)
        x = self.dec_6(x)
        x = torch.sigmoid(x)

        return x * input


if __name__ == "__main__":

    net = Net()
    x = torch.randn(1, 1, 512, 128)
    out = net(x)
    print(out.shape)
