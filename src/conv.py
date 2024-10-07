# Python 3.12

# PyTorch dependencies
import torch
import torch.nn as nn
from torch._tensor import Tensor

# Standard Library dependencies
from typing import Optional, Tuple

# Internal dependencies
from src.internal_types import _size_any_int
from src.utils import param_check, conv_param_check
from src.fold import Unfold

# recognice device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Conv(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, ...],
        stride: _size_any_int = 1,
        padding: _size_any_int = 0,
        dilation: _size_any_int = 1,
        bias: bool = True,
        input_size: Optional[Tuple[int, ...]] = None,
    ) -> None:

        # call super class constructor
        super(Conv, self).__init__()

        conv_param_check(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            input_size=input_size,
            bias=bias,
        )

        _kernel_size: Tuple[int, ...]
        _dilation: Tuple[int, ...]
        _padding: Tuple[int, ...]
        _stride: Tuple[int, ...]
        checked_params: Tuple[Tuple[int, ...], ...] = param_check(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        _kernel_size, _dilation, _padding, _stride = checked_params
        _kernel_size = tuple([input_channels, *_kernel_size])
        _dilation = tuple([1, *_dilation])
        _padding = tuple([0, *_padding])
        _stride = tuple([1, *_stride])

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, *_kernel_size, device=device)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, device=device)
        )

        # define submodules
        self.unfold: Tensor = Unfold(
            kernel_size=_kernel_size,
            dilation=_dilation,
            padding=_padding,
            stride=_stride,
            input_size=input_size,
        )

        # save contracting kernel dimensionality
        self.kernel_size: Tuple[int, ...] = _kernel_size
        self.use_bias: bool = bias

    def forward(self, input) -> Tensor:

        unfolded: Tensor = self.unfold(input)
        uN: int = unfolded.dim()
        kN: int = self.weight.dim()
        kC: int = len(self.kernel_size)  # number of kernel dimensions to contract
        bN: int = uN - 2 * kC  # number of batch dimensions
        oN: int = kN - kC  # number of output channel dimensions
        iN: int = bN + kC  # number of input dimensions

        # Note.
        # einsum performs better with many non-contracting elements
        # matmul performs better with many contracting elements

        output: Tensor
        non_contracting_elements: int = unfolded[uN - kC :].numel()
        contracting_elements: int = unfolded[: uN - kC].numel()
        if 100 * non_contracting_elements > contracting_elements:
            # einsum conctraction
            idxUnf: list[int] = list(range(uN))
            idxOCh: list[int] = list(range(uN, uN + oN))
            idxKer: list[int] = idxOCh + idxUnf[uN - kC :]
            idxOut: list[int] = idxUnf[:bN] + idxOCh + idxUnf[oN:iN]
            output = torch.einsum(unfolded, idxUnf, self.weight, idxKer, idxOut)
        else:
            # matmul contraction
            output = unfolded.flatten(uN - kC) @ self.weight.flatten(kN - kC).T
            permutation: Tuple[int, ...]
            permutation = tuple([*range(bN), *range(iN, iN + oN), *range(bN, iN)])
            output = output.permute(permutation).contiguous()
        # add the bias
        if self.use_bias:
            broadcast_size: Tuple[int, ...]
            broadcast_size = tuple([1] * bN + list(self.weight.shape[:oN]) + [1] * kC)
            output = output.add_(self.bias.view(broadcast_size))

        return output
