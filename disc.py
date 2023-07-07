import torch
import torch.nn as nn
from torchsummary import summary

class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Disc(nn.Module):
    def __init__(self, in_ch=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch*2, features[0], kernel_size=4, stride=2, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_ch = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_ch, feature, stride=1 if feature == features[-1] else 2)
            )
            in_ch = feature

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    model = Disc()
    preds = model(x, y)
    print(preds.shape)

    summary(model, [(3, 256, 256), (3, 256, 256)], device='cpu')


if __name__ == "__main__":
    test()