# Python 3.12

# PyTorch dependencies
import torch
import torch.nn.functional as F
from torch import Tensor

# Standard Library dependencies
import timeit
from typing import Tuple

# Internal dependencies
from src.conv import Conv


# Select device: use CUDA if available, else CPU
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def benchmark_convolutions_2d() -> None:
    """
    Benchmark the n-dimensional convolution (Conv) against PyTorch's 2D convolution.
    """
    # Define sizes (batch_size, channels, height, width)
    input_size: Tuple[int, ...] = (16, 3, 512, 512)
    kernel_size: Tuple[int, ...] = (3, 3)
    output_channels = 2

    # Create input tensor on the selected device
    input: Tensor = torch.arange(
        1,
        int(torch.prod(torch.Tensor(input_size))) + 1,
        dtype=torch.float32,
        device=device,
    ).view(input_size)

    # Create ConvND instance and move to device
    conv: Conv = Conv(
        input_channels=input_size[1],
        output_channels=output_channels,
        kernel_size=kernel_size,
        input_size=input_size,
    ).to(device)

    # Create weights and bias for conv2D on the device
    conv2d_weight: Tensor = torch.randn(
        output_channels, input_size[1], *kernel_size, dtype=torch.float32, device=device
    )
    conv2d_bias: Tensor = torch.randn(
        output_channels, dtype=torch.float32, device=device
    )

    # Set the same weights and bias to ConvND and conv2D
    conv.weight.data = conv2d_weight
    conv.bias.data = conv2d_bias

    # Define benchmark runs
    def run_convND_2d() -> None:
        conv(input)
        return None

    def run_conv2d() -> None:
        F.conv2d(input, weight=conv2d_weight, bias=conv2d_bias)
        return None

    # Define number of runs
    runs: int = 30

    # Synchronize CUDA before starting timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure convolution times for ConvND
    conv_time: float = timeit.timeit(run_convND_2d, number=runs) / runs

    # Synchronize CUDA before measuring PyTorch conv2d
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure convolution times for PyTorch conv2d
    torch_conv_time: float = timeit.timeit(run_conv2d, number=runs) / runs

    # Synchronize CUDA after timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    print()
    print(f"ConvND 2D time ({runs} runs): {conv_time:.5f} seconds")
    print(f"PyTorch conv2D time ({runs} runs): {torch_conv_time:.5f} seconds")
    print(f"ConvND/conv2D ratio: {conv_time / torch_conv_time:.5f}")

    return None


def benchmark_convolutions_3d() -> None:
    """
    Benchmark the n-dimensional convolution (Conv) against PyTorch's 3D convolution.
    """
    # Define sizes (batch_size, channels, depth, height, width)
    input_size: Tuple[int, ...] = (8, 3, 30, 64, 64)
    kernel_size: Tuple[int, ...] = (3, 3, 3)
    output_channels = 2

    # Create input tensor on the selected device
    input: Tensor = torch.arange(
        1,
        int(torch.prod(torch.Tensor(input_size))) + 1,
        dtype=torch.float32,
        device=device,
    ).view(input_size)

    # Create ConvND instance and move to device
    conv: Conv = Conv(
        input_channels=input_size[1],
        output_channels=output_channels,
        kernel_size=kernel_size,
        input_size=input_size,
    ).to(device)

    # Create weights and bias for conv3D on the device
    conv3d_weight: Tensor = torch.randn(
        output_channels, input_size[1], *kernel_size, dtype=torch.float32, device=device
    )
    conv3d_bias: Tensor = torch.randn(
        output_channels, dtype=torch.float32, device=device
    )

    # Set the same weights and bias to ConvND and conv3D
    conv.weight.data = conv3d_weight
    conv.bias.data = conv3d_bias

    # Define benchmark runs
    def run_convND() -> None:
        conv(input)
        return None

    def run_conv3d() -> None:
        F.conv3d(input, weight=conv3d_weight, bias=conv3d_bias)
        return None

    # Define number of runs
    runs: int = 30

    # Synchronize CUDA before starting timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure convolution times for ConvND
    conv_time: float = timeit.timeit(run_convND, number=runs) / runs

    # Synchronize CUDA before measuring PyTorch conv3d
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure convolution times for PyTorch conv3d
    torch_conv_time: float = timeit.timeit(run_conv3d, number=runs) / runs

    # Synchronize CUDA after timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    print()
    print(f"ConvND 3D time ({runs} runs): {conv_time:.5f} seconds")
    print(f"PyTorch conv3D time ({runs} runs): {torch_conv_time:.5f} seconds")
    print(f"ConvND/conv3D ratio: {conv_time / torch_conv_time:.5f}")

    return None


if __name__ == "__main__":
    benchmark_convolutions_2d()
    benchmark_convolutions_3d()
