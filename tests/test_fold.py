import pytest
import torch
from torch._tensor import Tensor
from torch.nn import Unfold as TorchUnfold
from torch.nn import Fold as TorchFold

from typing import Tuple

# Internal dependencies
from src.fold import Unfold, Fold

# recognice device
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.order(3)
def test_fold_without_padding() -> None:
    """
    Test the custom Fold implementation against PyTorch's Fold.
    """
    # Define sizes
    input_size: Tuple[int, ...] = (1, 2, 3, 4)
    kernel_size: Tuple[int, ...] = (2, 2)
    padding: Tuple[int, ...] = (0, 0)
    output_size: Tuple[int, ...] = input_size[-2:]

    # Create input tensor
    numel: int = 1
    for sz in input_size:
        numel *= sz
    input: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    input = input.view(size=input_size).to(device=device)

    # Custom Unfold and Fold
    unfold = Unfold(kernel_size=kernel_size, padding=padding)
    unfolded: Tensor = unfold(input)
    fold = Fold(output_size=output_size, kernel_size=kernel_size, padding=padding)
    folded_output1: Tensor = fold(unfolded)

    # PyTorch Unfold and Fold
    torch_unfold = TorchUnfold(kernel_size=kernel_size, padding=padding)
    unfolded_torch: Tensor = torch_unfold(input)
    torch_fold = TorchFold(
        output_size=output_size, kernel_size=kernel_size, padding=padding
    )
    folded_output2: Tensor = torch_fold(unfolded_torch)

    # Assert folded outputs are close
    assert torch.allclose(
        folded_output1, folded_output2, atol=1e-3
    ), "Fold outputs do not match"


@pytest.mark.order(4)
def test_fold_with_padding() -> None:
    """
    Test the custom Fold implementation against PyTorch's Fold.
    """
    # Define sizes
    input_size: Tuple[int, ...] = (1, 2, 3, 4)
    kernel_size: Tuple[int, ...] = (2, 2)
    padding: Tuple[int, ...] = (2, 1)
    output_size: Tuple[int, ...] = input_size[-2:]

    # Create input tensor
    numel: int = 1
    for sz in input_size:
        numel *= sz
    input: Tensor = torch.arange(start=1, end=numel + 1, dtype=torch.float32)
    input = input.view(size=input_size).to(device=device)

    # Custom Unfold and Fold
    unfold = Unfold(kernel_size=kernel_size, padding=padding)
    unfolded: Tensor = unfold(input)
    fold = Fold(output_size=output_size, kernel_size=kernel_size, padding=padding)
    folded_output1: Tensor = fold(unfolded)

    # PyTorch Unfold and Fold
    torch_unfold = TorchUnfold(kernel_size=kernel_size, padding=padding)
    unfolded_torch: Tensor = torch_unfold(input)
    torch_fold = TorchFold(
        output_size=output_size, kernel_size=kernel_size, padding=padding
    )
    folded_output2: Tensor = torch_fold(unfolded_torch)

    # Assert folded outputs are close
    assert torch.allclose(
        folded_output1, folded_output2, atol=1e-3
    ), "Fold outputs do not match"
