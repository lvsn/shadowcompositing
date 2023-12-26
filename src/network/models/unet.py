"""
Shadow Harmonization for Realisitc Compositing (c)
by Lucas Valença, Jinsong Zhang, Michaël Gharbi,
Yannick Hold-Geoffroy and Jean-François Lalonde.

Developed at Université Laval in collaboration with Adobe, for more
details please see <https://lvsn.github.io/shadowcompositing/>.

Work published at ACM SIGGRAPH Asia 2023. Full open access at the ACM
Digital Library, see <https://dl.acm.org/doi/10.1145/3610548.3618227>.

This code is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import torch.nn as nn

# Simple patch-based discriminator inspired by Pix2Pix and Deep Image Harmonization
class UNetDiscriminator(nn.Module):
    def __init__(self, in_res=128):
        super().__init__()
        encoder_modules = [ConvolutionBlock(3, 64),
                           ConvolutionBlock(64, 64),
                           ConvolutionBlock(64, 128),
                           ConvolutionBlock(128, 128, in_res=False, activation=False)]
        self.encoder = nn.ModuleList(encoder_modules)
        self.output = nn.Sigmoid()

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        x = self.output(x)
        return x

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1,
                 in_res=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)]
        if in_res: layers.append(nn.InstanceNorm2d(out_channels))
        if activation: layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
