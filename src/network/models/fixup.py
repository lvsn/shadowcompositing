"""
Code modified from UnetFixUp, see <https://github.com/Jovp/fixupunet>.
Publication "OutCast: Single Image Relighting with Cast Shadows,"
Computer Graphics Forum, 2022, by Griffiths et al.

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

from __future__ import absolute_import

from torch import nn
import torch
import numpy as np
from collections import OrderedDict
from omegaconf import DictConfig
import itertools

class FixUpUnet(nn.Module):
    def __init__(self, in_res=128, detection_input=False):
        super(FixUpUnet, self).__init__()

        switch = {64: (4, 128),
                  128: (5, 256),
                  256: (6, 512),
                  512: (7, 1024),
                  1024: (8, 2048)}

        down_layers = switch[in_res][0]
        max_feat = switch[in_res][1]

        in_feat = 4 if not detection_input else 5

        cfg = DictConfig({'feat': 32,
                          'in_feat': in_feat,
                          'out_feat': 4,
                          'down_layers': down_layers,
                          'identity_layers': 3,
                          'bottleneck_layers': 6,
                          'skips': True,
                          'act_fn': 'relu',
                          'out_act_fn': 'sigmoid',
                          'max_feat': max_feat,
                          'script_submodules': False})

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1

        layer = FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, "none", cfg.act_fn)
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.in_conv = layer

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            # Residual chain
            layer = FixupResidualChain(
                feat_curr,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

            # Downsampling convolution
            layer = FixupConvModule(
                feat_curr, feat_next, 4, 2, True, "none", "none", use_bias=True
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.down_layers.append(layer)

        self.bottleneck_layers = nn.ModuleList()
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        layer = FixupResidualChain(
            feat_curr,
            cfg.bottleneck_layers,
            3,
            cfg.act_fn,
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.bottleneck_layers.append(layer)

        self.up_layers = nn.ModuleList()
        for i in range(cfg.down_layers, 0, -1):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i - 1) * feat, max_feat)
            # Upsample
            self.up_layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            # Merge skip and upsample
            if self.skip:
                layer = FixupConvModule(
                    feat_next + feat_curr,
                    feat_next,
                    1,
                    1,
                    False,
                    "none",
                    "none",
                    use_bias=True,
                )
                if cfg.script_submodules:
                    layer = torch.jit.script(layer)
                self.up_layers.append(layer)
            # Residual chain
            layer = FixupResidualChain(
                feat_next,
                cfg.identity_layers,
                3,
                cfg.act_fn,
                depth_init=2 * cfg.identity_layers,
                single_padding=(i - 1 < 3),
            )
            if cfg.script_submodules:
                layer = torch.jit.script(layer)
            self.up_layers.append(layer)

        layer = FixupConvModule(
            feat, cfg.out_feat, 3, 1, True, "none", cfg.out_act_fn, use_bias=True
        )
        if cfg.script_submodules:
            layer = torch.jit.script(layer)
        self.out_conv = layer

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if i % 2 == 0:
                skips.append(x)

        for layer in self.bottleneck_layers:
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x)

            if self.skip:
                if i % 3 == 0:
                    x = torch.cat([x, skips.pop()], dim=1)

        return self.out_conv(x)









act_fn_map = {
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(inplace=False),
    "leaky_relu": nn.LeakyReLU(0.2, inplace=False),
    "tanh": nn.Tanh(),
    "none": nn.Identity(),
}

norm_map_2d = {
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d,
    "none": None,
}


class FixupConvModule(nn.Module):
    """Basic convolution module with conv + norm(optional) + activation(optional).

    Args:
      n_in(int): number of input channels.
      n_out(int): number of output channels.
      ksize(int): size of the convolution kernel (square).
      stride(int): downsampling factor
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(
        self,
        n_in,
        n_out,
        ksize=3,
        stride=1,
        pad=True,
        norm_layer="none",
        activation="none",
        padding_mode="reflect",
        use_bias=False,
        sn=False,
    ):
        super(FixupConvModule, self).__init__()

        assert (
            isinstance(n_in, int) and n_in > 0
        ), "Input channels should be a positive integer got {}".format(n_in)
        assert (
            isinstance(n_out, int) and n_out > 0
        ), "Output channels should be a positive integer got {}".format(n_out)
        assert (
            isinstance(ksize, int) and ksize > 0
        ), "Kernel size should be a positive integer got {}".format(ksize)

        layers = OrderedDict()

        padding = (ksize - 1) // 2 if pad else 0
        padding_mode = padding_mode if pad else "zeros"

        if not sn:
            layers["conv"] = nn.Conv2d(
                n_in,
                n_out,
                ksize,
                stride=stride,
                padding=padding,
                bias=use_bias,
                padding_mode=padding_mode,
            )
        else:
            layers["conv"] = torch.nn.utils.spectral_norm(
                nn.Conv2d(
                    n_in,
                    n_out,
                    ksize,
                    stride=stride,
                    padding=padding,
                    bias=use_bias,
                    padding_mode=padding_mode,
                )
            )

        if norm_layer != "none":
            layers["norm"] = norm_map_2d[norm_layer](n_out)

        if activation != "none":
            layers["activation"] = act_fn_map[activation]

        # Initialize parameters
        _init_fc_or_conv(layers["conv"], activation)

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = self.net(x)
        return x


class ConvChain(nn.Module):
    """Linear chain of convolution layers.

    Args:
      n_in(int): number of input channels.
      ksize(int or list of int): size of the convolution kernel (square).
      width(int or list of int): number of features channels in the intermediate layers.
      depth(int): number of layers
      strides(list of int): stride between kernels. If None, defaults to 1 for all.
      pad(bool): if True, zero pad the convolutions to maintain a constant size.
      activation(str): nonlinear activation function between convolutions.
      norm_layer(str): normalization to apply between the convolution modules.
    """

    def __init__(
        self,
        n_in,
        ksize=3,
        width=64,
        depth=3,
        strides=None,
        pad=True,
        activation="relu",
        norm_layer=None,
        padding_mode="reflect",
    ):
        super(ConvChain, self).__init__()

        assert (
            isinstance(n_in, int) and n_in > 0
        ), "Input channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list
        ), "Kernel size should be a positive integer or a list of integers"
        assert (
            isinstance(depth, int) and depth > 0
        ), "Depth should be a positive integer"
        assert isinstance(width, int) or isinstance(
            width, list
        ), "Width should be a list or an int"

        _in = [n_in]

        if strides is None:
            _strides = [1] * depth
        else:
            assert isinstance(strides, list), "strides should be a list"
            assert len(strides) == depth, "strides should have `depth` elements"
            _strides = strides

        if isinstance(width, int):
            _in = _in + [width] * (depth - 1)
            _out = [width] * depth
        elif isinstance(width, list):
            assert (
                len(width) == depth
            ), "Specifying width with a list should have `depth` elements"
            _in = _in + width[:-1]
            _out = width

        if isinstance(ksize, int):
            _ksizes = [ksize] * depth
        elif isinstance(ksize, list):
            assert len(ksize) == depth, "kernel size list should have 'depth' entries"
            _ksizes = ksize

        _activations = [activation] * depth
        _padding_modes = [padding_mode] * depth
        # dont normalize in/out layers
        _norms = [norm_layer] * depth

        # Core processing layers, no norm at the first layer
        layers = OrderedDict()
        for lvl in range(depth):
            layers["conv{}".format(lvl)] = FixupConvModule(
                _in[lvl],
                _out[lvl],
                _ksizes[lvl],
                stride=_strides[lvl],
                pad=pad,
                activation=_activations[lvl],
                norm_layer=_norms[lvl],
                padding_mode=_padding_modes[lvl],
                use_bias=False,
            )

        self.net = nn.Sequential(layers)

    def forward(self, x):
        x = self.net(x)
        return x


@torch.jit.script
def add_v1(x, y):
    b, c, h, w = x.shape
    x = x.view(-1) + y
    return x.view(b, c, h, w)


class FixupBasicBlock(nn.Module):
    # https://openreview.net/pdf?id=H1gsz30cKX
    expansion = 1

    def __init__(
        self,
        n_features,
        ksize=3,
        padding=True,
        padding_mode="reflect",
        activation="relu",
        activation2=None,
    ):
        super(FixupBasicBlock, self).__init__()

        self.padding = padding
        self.hks = (ksize - 1) // 2

        if activation2 is None:
            activation2 = activation

        self.conv1 = torch.jit.script(
            FixupConvModule(
                n_features,
                n_features,
                ksize=ksize,
                stride=1,
                pad=padding,
                activation="none",
                norm_layer="none",
                padding_mode=padding_mode,
            )
        )

        self.activation = act_fn_map[activation]
        if activation2 is None:
            activation2 = activation
        self.activation2 = act_fn_map[activation2]

        self.conv2 = torch.jit.script(
            FixupConvModule(
                n_features,
                n_features,
                ksize=ksize,
                stride=1,
                pad=padding,
                activation="none",
                norm_layer="none",
                padding_mode=padding_mode,
            )
        )

        self.bias0 = torch.nn.Parameter(torch.tensor(0.0))
        self.bias1 = torch.nn.Parameter(torch.tensor(0.0))
        self.bias2 = torch.nn.Parameter(torch.tensor(0.0))
        self.bias3 = torch.nn.Parameter(torch.tensor(0.0))
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.padding:
            identity = x
        else:
            identity = x[
                ..., 2 * self.hks : -2 * self.hks, 2 * self.hks : -2 * self.hks
            ]
        out = x + self.bias0
        out = self.conv1(out)
        out = out + self.bias1
        out = self.activation(out)
        out = out + self.bias2
        out = self.conv2(out)
        out = out * self.scale
        out = self.bias3 + out
        out = identity + out
        out = self.activation2(out)
        return out


class FixupResidualChain(nn.Module):
    """Linear chain of residual blocks.
    Args:
      n_features(int): number of input channels.
      ksize(int): size of the convolution kernel (square).
      depth(int): number of residual blocks
      activation(str): nonlinear activation function between convolutions.
    """

    def __init__(
        self,
        n_features,
        depth=3,
        ksize=3,
        activation="relu",
        last_activation="relu",
        padding_mode="reflect",
        depth_init=None,
        single_padding=False,
    ):
        super(FixupResidualChain, self).__init__()

        assert (
            isinstance(n_features, int) and n_features > 0
        ), "Number of feature channels should be a positive integer"
        assert (isinstance(ksize, int) and ksize > 0) or isinstance(
            ksize, list
        ), "Kernel size should be a positive integer or a list of integers"
        assert (
            isinstance(depth, int) and depth > 0 and depth < 16
        ), "Depth should be a positive integer lower than 16"

        self.depth = depth
        if depth_init is not None:
            self.depth_init = depth_init
        else:
            self.depth_init = self.depth

        # Core processing layers
        common_layers = OrderedDict()

        if single_padding:
            hks = (ksize - 1) // 2
            p = hks * 2 * depth
            common_layers["early_pad"] = torch.nn.ReflectionPad2d(padding=(p, p, p, p))
        for lvl in range(0, depth):

            if lvl == depth - 1:
                activation2 = last_activation
            else:
                activation2 = activation

            blockname = "resblock{}".format(lvl)
            common_layers[blockname] = torch.jit.script(
                FixupBasicBlock(
                    n_features,
                    ksize=ksize,
                    activation=activation,
                    activation2=activation2,
                    padding_mode=padding_mode,
                    padding=not single_padding,
                )
            )

        self.com_net = nn.Sequential(common_layers)

        self._reset_weights()

    def _reset_weights(self):
        for m in itertools.chain(self.com_net.modules()):
            if isinstance(m, FixupBasicBlock) or (
                isinstance(m, torch.jit.RecursiveScriptModule)
                and m.original_name == "FixupBasicBlock"
            ):
                nn.init.normal_(
                    m.conv1.net.conv.weight,
                    mean=0,
                    std=np.sqrt(
                        2
                        / (
                            m.conv1.net.conv.weight.shape[0]
                            * np.prod(m.conv1.net.conv.weight.shape[2:])
                        )
                    )
                    * self.depth_init ** (-0.5),
                )
                nn.init.constant_(m.conv2.net.conv.weight, 0)

    def forward(self, x):
        x = self.com_net(x)
        return x


def _init_fc_or_conv(fc_conv, activation):
    gain = 1.0
    if activation != "none":
        try:
            gain = nn.init.calculate_gain(activation)
        except:
            print("Warning using gain of ", gain, " for activation: ", activation)
    nn.init.xavier_uniform_(fc_conv.weight, gain)
    if fc_conv.bias is not None:
        nn.init.constant_(fc_conv.bias, 0.0)
