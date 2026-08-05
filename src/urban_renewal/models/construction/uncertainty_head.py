from torch import nn


class UncertaintyHead(nn.Sequential):
    def __init__(self, channels: int):
        super().__init__(nn.Conv2d(channels, channels // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels // 2, 1, 1))
