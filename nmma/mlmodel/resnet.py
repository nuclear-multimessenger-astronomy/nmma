from typing import Callable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ChannelNorm(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_groups: Optional[int] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        num_groups = num_groups or num_channels
        if num_channels % num_groups:
            raise ValueError("num_groups must be a factor of num_channels")

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.channels_per_group = self.num_channels // self.num_groups
        self.eps = eps

        shape = (self.num_channels, 1)
        self.weight = torch.nn.Parameter(torch.ones(shape))
        self.bias = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        keepdims = self.num_groups == self.num_channels

        # compute group variance via the E[x**2] - E**2[x] trick
        mean = x.mean(-1, keepdims=keepdims)
        sq_mean = (x**2).mean(-1, keepdims=keepdims)

        # if we have groups, do some reshape magic
        # to calculate group level stats then
        # reshape back to full channel dimension
        if self.num_groups != self.num_channels:
            mean = torch.stack([mean, sq_mean], dim=1)
            mean = mean.reshape(-1, 2, self.num_groups, self.channels_per_group)
            mean = mean.mean(-1, keepdims=True)
            mean = mean.expand(-1, -1, -1, self.channels_per_group)
            mean = mean.reshape(-1, 2, self.num_channels, 1)
            mean, sq_mean = mean[:, 0], mean[:, 1]

        # roll the mean and variance into the
        # weight and bias so that we have to do
        # fewer computations along the full time axis
        std = (sq_mean - mean**2 + self.eps) ** 0.5
        scale = self.weight / std
        shift = self.bias - scale * mean
        return shift + x * scale


def get_norm_layer(groups: Optional[int] = None) -> nn.Module:
    class GroupNorm(ChannelNorm):
        def __init__(self, num_channels: int) -> None:
            num_groups = None if groups is None else min(num_channels, groups)
            super().__init__(num_channels, num_groups)

    return GroupNorm


def convN(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv1d:
    """1d convolution with padding"""
    if not kernel_size % 2:
        raise ValueError("Can't use even sized kernels")

    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * int(kernel_size // 2),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """kernel-size 1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = convN(inplanes, planes, kernel_size, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convN(planes, planes, kernel_size)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck blocks implement one extra convolution
    compared to basic blocks. In this layers, the `planes`
    parameter is generally meant to _downsize_ the number
    of feature maps first, which then get expanded out to
    `planes * Bottleneck.expansion` feature maps at the
    output of the layer.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        width = int(planes * (base_width / 64.0)) * groups

        # conv1 does no downsampling, just reduces the number of
        # feature maps from inplanes to width (where width == planes)
        # if groups == 1 and base_width == 64
        self.conv1 = convN(inplanes, width, kernel_size)
        self.bn1 = norm_layer(width)

        # conv2 keeps the same number of feature maps,
        # but downsamples along the time axis if stride
        # or dilation > 1
        self.conv2 = convN(width, width, kernel_size, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        # conv3 expands the feature maps back out to planes * expansion
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """1D ResNet architecture

    Simple extension of ResNet to 1D convolutions with
    arbitrary kernel sizes to support the longer timeseries
    used in BBH detection.

    Args:
        num_ifos:
            The number of interferometers used for BBH
            detection. Sets the channel dimension of the
            input tensor
        layers:
            A list representing the number of residual
            blocks to include in each "layer" of the
            network. Total layers (e.g. 50 in ResNet50)
            is `2 + sum(layers) * factor`, where factor
            is `2` for vanilla `ResNet` and `3` for
            `BottleneckResNet`.
        kernel_size:
            The size of the convolutional kernel to
            use in all residual layers. _NOT_ the size
            of the input kernel to the network, which
            is determined at run-time.
        zero_init_residual:
            Flag indicating whether to initialize the
            weights of the batch-norm layer in each block
            to 0 so that residuals are initialized as
            identities. Can improve training results.
        groups:
            Number of convolutional groups to use in all
            layers. Grouped convolutions induce local
            connections between feature maps at subsequent
            layers rather than global. Generally won't
            need this to be >1, and wil raise an error if
            >1 when using vanilla `ResNet`.
        width_per_group:
            Base width of each of the feature map groups,
            which is scaled up by the typical expansion
            factor at each layer of the network. Meaningless
            for vanilla `ResNet`.
        stride_type:
            Whether to achieve downsampling on the time axis
            by strided or dilated convolutions for each layer.
            If left as `None`, strided convolutions will be
            used at each layer. Otherwise, `stride_type` should
            be one element shorter than `layers` and indicate either
            `stride` or `dilation` for each layer after the first.
        norm_groups:
            The number of groups to use in GroupNorm layers
            throughout the model. If left as `-1`, the number
            of groups will be equal to the number of channels,
            making this equilavent to LayerNorm
    """

    block = BasicBlock

    def __init__(
        self,
        num_ifos: Tuple[int, int],
        context_dim: int,
        layers: List[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        # TODO: use Literal["stride", "dilation"] once typeo fix is in
        stride_type: Optional[List[str]] = None,
        norm_groups: Optional[int] = None,
    ) -> None:
        super().__init__()

        # hack since other embedding require both n_ifos and strain
        # to initialize
        num_ifos, _ = num_ifos
        self._norm_layer = get_norm_layer(norm_groups)

        self.inplanes = 64
        self.dilation = 1

        # TODO: should we support passing a single string
        # for simplicity here?
        if stride_type is None:
            # each element in the tuple indicates if we should replace
            # the stride with a dilated convolution instead
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(
                "'stride_type' should be None or a "
                "{}-element tuple, got {}".format(len(layers) - 1, stride_type)
            )

        self.groups = groups
        self.base_width = width_per_group

        # start with a basic conv-bn-relu-maxpool block
        # to reduce the dimensionality before the heavy
        # lifting starts
        self.conv1 = nn.Conv1d(
            num_ifos,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # now create layers of residual blocks where each
        # layer uses the same number of feature maps for
        # all its blocks (some power of 2 times 64).
        # Don't downsample along the time axis in the first
        # layer, but downsample in all the rest (either by
        # striding or dilating depending on the stride_type
        # argument)
        residual_layers = [self._make_layer(64, layers[0], kernel_size)]
        it = zip(layers[1:], stride_type)
        for i, (num_blocks, stride) in enumerate(it):
            block_size = 64 * 2 ** (i + 1)
            layer = self._make_layer(
                block_size,
                num_blocks,
                kernel_size,
                stride=2,
                stride_type=stride,
            )
            residual_layers.append(layer)
        self.residual_layers = nn.ModuleList(residual_layers)

        # Average pool over each feature map to create a
        # single value for each feature map that we'll use
        # in the fully connected head
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # use a fully connected layer to map from the
        # feature maps to the binary output that we need
        self.fc = nn.Linear(block_size * self.block.expansion, context_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        kernel_size: int = 3,
        stride: int = 1,
        stride_type: Literal["stride", "dilation"] = "stride",
    ) -> nn.Sequential:
        block = self.block
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride_type == "dilation":
            self.dilation *= stride
            stride = 1
        elif stride_type != "stride":
            raise ValueError("Unknown stride type {stride}")

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel_size,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.residual_layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# TODO: implement as arg of ResNet instead?
class BottleneckResNet(ResNet):
    """A version of ResNet that uses bottleneck blocks"""

    block = Bottleneck
