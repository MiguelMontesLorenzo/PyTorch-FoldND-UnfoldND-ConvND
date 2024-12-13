# Pytorch dependencies
import torch
from torch import Tensor
from torch.nn import Unfold as TorchUnfold

# Standard Library dependencies
from typing import Tuple

# Other 3rd party dependencies
import pytest

# Internal dependencies
from src.fold import Unfold

# recognice device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.order(1)
def test_unfold_without_padding() -> None:
    """
    Test the custom Unfold implementation against PyTorch's Unfold.
    """
    # Define sizes
    input_size: Tuple[int, ...] = (1, 1, 3, 4)
    kernel_size: Tuple[int, ...] = (2, 2)
    padding: Tuple[int, ...] = (0, 0)

    # Create input tensor
    numel: int = 1
    for sz in input_size:
        numel *= sz
    input: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    input = input.view(size=input_size).to(device=device)

    # Create kernel tensor
    numel = 1
    for sz in kernel_size:
        numel *= sz
    kernel: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    kernel = kernel.view(size=kernel_size).to(device=device)

    # Custom Unfold
    unfold1 = Unfold(kernel_size=kernel_size, padding=padding)
    unfolded1: Tensor = unfold1(input)
    output1: Tensor = torch.einsum("eijkhq,hq->eijk", unfolded1, kernel).squeeze()

    # PyTorch Unfold
    unfold2 = TorchUnfold(kernel_size=kernel_size, padding=padding)
    unfolded2: Tensor = unfold2(input)
    output2: Tensor = torch.einsum("ijk,j->ik", unfolded2, kernel.flatten()).squeeze()

    # Assert outputs are close
    assert torch.allclose(
        output1.flatten(), output2.flatten(), atol=1e-3
    ), "Unfold outputs do not match"


@pytest.mark.order(2)
def test_unfold_with_padding() -> None:
    """
    Test the custom Unfold implementation against PyTorch's Unfold.
    """
    # Define sizes
    input_size: Tuple[int, ...] = (1, 1, 3, 4)
    kernel_size: Tuple[int, ...] = (2, 2)
    padding: Tuple[int, ...] = (2, 1)

    # Create input tensor
    numel: int = 1
    for sz in input_size:
        numel *= sz
    input: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    input = input.view(size=input_size).to(device=device)

    # Create kernel tensor
    numel = 1
    for sz in kernel_size:
        numel *= sz
    kernel: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    kernel = kernel.view(size=kernel_size).to(device=device)

    # Custom Unfold
    unfold1 = Unfold(kernel_size=kernel_size, padding=padding)
    unfolded1: Tensor = unfold1(input)
    output1: Tensor = torch.einsum("eijkhq,hq->eijk", unfolded1, kernel).squeeze()

    # PyTorch Unfold
    unfold2 = TorchUnfold(kernel_size=kernel_size, padding=padding)
    unfolded2: Tensor = unfold2(input)
    output2: Tensor = torch.einsum("ijk,j->ik", unfolded2, kernel.flatten()).squeeze()

    # Assert outputs are close
    assert torch.allclose(
        output1.flatten(), output2.flatten(), atol=1e-3
    ), "Unfold outputs do not match"
