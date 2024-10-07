import pytest
import torch
import torch.nn.functional as F
from torch._tensor import Tensor

# Standard library dependencies
import math
from typing import Tuple

# Internal dependencies
from src.conv import Conv

# recognice device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.order(5)
def test_convND_2D() -> None:
    """
    Test the custom Conv implementation for 2D convolutions against PyTorch's conv2d.
    """
    # Define sizes
    input_size: Tuple[int, ...] = (1, 3, 5, 5)  # (batch_size, channels, height, width)
    kernel_size: Tuple[int, ...] = (3, 3)  # (height, width)
    output_channels = 2

    # Create input tensor
    numel: int = int(math.prod(input_size)) + 1
    input: Tensor = torch.arange(1, numel, dtype=torch.float32).view(size=input_size)
    input = input.to(device=device)

    # Initialize custom ConvND
    conv = Conv(
        input_channels=input_size[1],
        output_channels=output_channels,
        kernel_size=kernel_size,
    )

    # Create weights and bias for conv2D
    conv2d_weight: Tensor = torch.randn(
        output_channels, input_size[1], *kernel_size, dtype=torch.float32
    )
    conv2d_bias: Tensor = torch.randn(output_channels, dtype=torch.float32)
    conv2d_weight = conv2d_weight.to(device=device)
    conv2d_bias = conv2d_bias.to(device=device)

    # Set weights and bias
    conv.weight.data = conv2d_weight
    conv.bias.data = conv2d_bias

    # Custom ConvND forward pass
    output1: Tensor = conv(input)

    # PyTorch conv2d forward pass
    output2: Tensor = F.conv2d(input, weight=conv2d_weight, bias=conv2d_bias)

    # Assert outputs are close
    assert torch.allclose(
        output1.squeeze().flatten(), output2.squeeze().flatten(), atol=1e-3
    ), "2D convolution outputs do not match"


@pytest.mark.order(6)
def test_convND_3D() -> None:
    """
    Test the custom Conv implementation for 3D convolutions against PyTorch's conv3d.
    """
    # Define sizes (batch_size, channels, depth, height, width)
    input_size: Tuple[int, ...] = (1, 3, 5, 5, 5)
    kernel_size: Tuple[int, ...] = (3, 3, 3)  # (depth, height, width)
    output_channels = 2

    # Create input tensor
    numel: int = int(math.prod(input_size)) + 1
    input: Tensor = torch.arange(1, numel, dtype=torch.float32).view(size=input_size)
    input = input.to(device=device)

    # Initialize custom ConvND
    convND: Conv = Conv(
        input_channels=input_size[1],
        output_channels=output_channels,
        kernel_size=kernel_size,
    )

    # Create weights and bias for conv3D
    conv3d_weight: Tensor = torch.randn(
        output_channels, input_size[1], *kernel_size, dtype=torch.float32
    )
    conv3d_bias: Tensor = torch.randn(output_channels, dtype=torch.float32)
    conv3d_weight = conv3d_weight.to(device=device)
    conv3d_bias = conv3d_bias.to(device=device)

    # Set weights and bias
    convND.weight.data = conv3d_weight
    convND.bias.data = conv3d_bias

    # Custom ConvND forward pass
    output1: Tensor = convND(input)

    # PyTorch conv3d forward pass
    output2: Tensor = F.conv3d(input, weight=conv3d_weight, bias=conv3d_bias)

    # Assert outputs are close
    assert torch.allclose(
        output1.squeeze().flatten(), output2.squeeze().flatten(), atol=1e-3
    ), "3D convolution outputs do not match"
