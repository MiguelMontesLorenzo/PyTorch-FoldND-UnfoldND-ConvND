# Python 3.12

# PyTorch dependencies
import torch
import torch.nn as nn
from torch._tensor import Tensor
from torch.nn.functional import pad

# Standard Library dependencies
import math
from typing import Callable, Optional, Tuple, Union

# Internal dependencies
from src.internal_types import _size_any_int
from src.utils import (
    param_check,
    fold_input_check,
    fold_output_check,
    unfold_input_check,
)


class Fold(nn.Module):
    """
    A class that implements an n-dimensional fold operation, reversing the effect of
    an unfolding operation by reconstructing the original input from its unfolded
    representation. This generalizes the folding process to n-dimensions, unlike
    PyTorch's native `torch.nn.Fold`, which is limited to 2D operations. This
    implementation provides flexibility in how input dimensions are managed during
    the folding process.

    The input format can be controlled via the `kernel_position` argument:

        - If `kernel_position == "last"` (the default case):
            The input dimensions are expected to follow this order:
            (*batch_dims, *conv_output_dims, *kernel_dims)

        - If `kernel_position == "first"`:
            The input dimensions are expected to follow this order:
            (*batch_dims, *kernel_dims, *conv_output_dims)

    This flexibility allows the class to handle different input formats based on how
    the kernel and convolutional output dimensions are arranged in the unfolded input.

    This differs from PyTorch's native 2D fold operation, which typically handles input
    in the shape:

        (N, C * kernel_height * kernel_width, L)

    where L is the number of sliding windows per convolution input.

    Additionally, this class allows an optional `input_size` argument to be passed
    during initialization. If `input_size` is provided, it should correspond to the
    non-batched convolution input size (i.e., excluding batch dimensions). If batch
    dimensions are included, they will be ignored. By providing `input_size`, certain
    calculations can be performed during initialization, potentially accelerating the
    `forward` method by avoiding repetitive computation. Note that the convolution input
    size is equivalent to fold function output size.
    """

    def __init__(
        self,
        kernel_size: _size_any_int,
        dilation: _size_any_int = 1,
        padding: _size_any_int = 0,
        stride: _size_any_int = 1,
        input_size: Optional[_size_any_int] = None,
        output_size: Optional[_size_any_int] = None,
        kernel_position: str = "last",
    ) -> None:
        """
        Args:
            kernel_size (int or tuple of int):
                The size of the sliding window for each spatial dimension. If an integer
                is provided, the same value is used for all spatial dimensions.
            dilation (int or tuple of int, optional):
                The spacing between kernel elements along each spatial dimension.
                Default is 1 (no dilation).
            padding (int or tuple of int, optional):
                The amount of padding added to each side of the input along each spatial
                dimension. Default is 0 (no padding).
            stride (int or tuple of int, optional):
                The step size for the sliding window along each spatial dimension.
                Default is 1 (no stride).
            input_size (Optional[int or tuple of int], optional):
                The size of the input tensor (excluding batch dimensions). If provided,
                this allows for pre-calculations during initialization that can speed up
                the `forward` method. Batch dimensions will be ignored. If not provided,
                these calculations will be performed during the `forward` method.
            output_size (Optional[int or tuple of int], optional):
                The expected output size of the tensor after the folding operation
                (excluding batch dimensions). If `input_size` is provided, `output_size`
                is validated during initialization to ensure consistency with the input
                dimensions and folding parameters.
            kernel_position (str, optional):
                Controls the position of the kernel dimensions in the input tensor.
                If `kernel_position == "last"` (default), input dimensions are expected
                in the order (*batch_dims, *conv_output_dims, *kernel_dims). If
                `kernel_position == "first"`, input dimensions should follow the order
                (*batch_dims, *kernel_dims, *conv_output_dims). This flexibility allows
                the class to handle different input formats depending on how the kernel
                and convolutional output dimensions are arranged.
        """

        super(Fold, self).__init__()  # This initializes the base Module class

        self.kernel_size: Tuple[int, ...]
        self.dilation: Tuple[int, ...]
        self.padding: Tuple[int, ...]
        self.stride: Tuple[int, ...]
        checked_params: Tuple[Tuple[int, ...], ...] = param_check(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        self.kernel_size, self.dilation, self.padding, self.stride = checked_params
        self.mask: Tensor
        self.unfold_size: Tuple[int, ...]
        self.indices: Optional[Tuple[Tensor, ...]] = None
        self.input_size: Optional[Tuple[int, ...]] = None
        self.output_size: Optional[Tuple[int, ...]] = None
        self.kernel_position: str = "last"
        self.input_size = fold_input_check(
            input_size, self.kernel_size, self.kernel_position
        )
        if input_size is not None:
            if output_size is not None:
                self.output_size = fold_output_check(
                    input_size=input_size, output_size=output_size, *checked_params
                )
            self.build_mask()
        self.indices: Optional[Tuple[Tensor, ...]] = None
        self.expanded_indices = None
        if not isinstance(kernel_position, str):
            raise TypeError(
                f"Incorrect type for kernel_position: "
                f"got {type(kernel_position).__name__}, expected str."
            )
        if kernel_position not in ["first", "last"]:
            raise ValueError(
                f"Incorrect value for kernel_position: "
                f"got '{kernel_position}', "
                f"expected one of: ['first', 'last']."
            )
        else:
            self.kernel_position = kernel_position

        return None

    def build_mask(self) -> None:
        """
        Creates a mask indicating the regions of the input tensor from which elements
        will be selected during the unfolding process. The mask has dimensions:

            (*input_dims, *kernel_element_windows)

        where `kernel_element_windows` represents the projections of each kernel element
        over the input feature space. This is conceptually similar to `kernel windows`,
        but using `kernel element windows` reduces the number of windows per dimension
        from approximately the input dimension size to approximately the kernel
        dimension size.

        The method constructs this mask by considering the padded input size, the
        kernel size, dilation, and stride values, applying these operations to mark the
        regions of the input that correspond to the different windows defined by the
        kernel’s elements.

        Steps:
        1. **Measure dimensionalities**: Determines the number of dimensions in both the
        input and the kernel.
        2. **Compute padded input size**: Adjusts the input size to account for any
        padding applied to the input.
        3. **Define dilated kernel size**: Calculates the effective kernel size after
        applying dilation to the kernel elements.
        4. **Account for mask size**: Initializes the mask with the appropriate size
        based on the padded input dimensions.
        5. **Generate kernel[0,...,0] mask**: Creates a mask for the first kernel
        element by projecting it over the input feature space. This mask is then
        modified to reflect the stride.
        6. **Iterate over kernel dimensions**: For each dimension, the method expands
        the mask and re-indexes the input to account for dilation and stride, repeating
        the process along each kernel dimension.
        7. **Save the final mask**: After iteration, the mask is saved as a class
        attribute, along with the `unfold_size`, which is used during the unfolding
        process.

        The generated mask is used internally by the `Unfold` class to apply the unfold
        operation across n-dimensions, ensuring that only the elements corresponding to
        the kernel windows are selected during unfolding.

        No value is returned, but the mask and unfold size are stored as class
        attributes: (`self.mask` and `self.unfold_size`).

        Note. The input of a convolution corresponds to the output of the fold function.
        Is because of this that along the code the input and output denominations might
        appear swaped.
        """

        # meassure dimensionalities (input, kernel, batch)
        kN: int = len(self.kernel_size)
        iN: int = len(self.input_size)
        bN: int = iN - 2 * kN
        # calculate padded output size and output size
        pre_size: Tuple[int, ...] = self.input_size[bN : iN - kN]
        aux_zp: zip = zip(pre_size, self.kernel_size, self.stride, self.dilation)
        pdd_size: Tuple[int, ...]
        pdd_size = tuple([s * (i - 1) + d * (k - 1) + 1 for i, k, s, d in aux_zp])

        # define dilated kernel size from kernel size and dilation
        dil_size: Callable[[int, int], int] = lambda k, d: k + (k - 1) * (d - 1)
        # kernel[0,...,0] element projection mask over input feature space
        knl0mask: Tensor = torch.full(size=pdd_size, fill_value=False)
        zp: zip = zip(pdd_size[bN:], self.kernel_size, self.stride, self.dilation)
        lzp: list[Tuple[int, ...]] = list(zp)
        knl0mask[[slice(i - dil_size(k, d) + 1) for i, k, _, d in lzp]].logical_not_()
        # apply stride to the kernel[0,...,0] mask
        on_stride: Tensor = torch.full_like(input=knl0mask, fill_value=False)
        on_stride[tuple(slice(0, i, s) for i, _, s, _ in lzp)] = True
        knl0mask.logical_and_(on_stride)

        # initialize variables for the loop
        knl_mask: Tensor = knl0mask
        reps: list[int] = []
        ms: Tuple[int, ...] = pdd_size[bN:]  # to account mask size
        # iterate over kernel dimensions repeating n each along a new dimension
        for i, (si, sk, _, sd) in enumerate(lzp):
            # kernel window adjusted by dilation
            sk: int = dil_size(sk, sd)
            # compute auxiliar step size & next mask size
            kdms: int = len(ms) - kN
            auxstp: list[Union[slice, None]]
            auxstp = [slice(ms[j + kdms]) if j == i else None for j in range(kN)]
            auxstp = [None] * (kdms) + [slice(sk)] + auxstp
            resize: list[int] = list(ms[:kdms]) + [sk] + list(ms[kdms:])
            # compute reindexation indices
            shift: Tensor = torch.arange(start=0, end=(sk * sd), step=sd)[:, None]
            reidx: Tensor
            reidx = (torch.arange(start=0, end=si)[None, :] - shift) % si
            reidx = reidx[auxstp]  # aux step to enssure proper dim assignment
            reidx = reidx.expand(resize)
            # reindex
            knl_mask = knl_mask.unsqueeze(kdms).expand(resize)
            knl_mask = knl_mask.gather(i + kdms + 1, reidx)
            # iteration saves
            ms = tuple(knl_mask.shape)
            reps.append(sk)

        # define class attributes
        self.mask: Tensor = knl_mask
        self.padded_size: Tuple[int, ...] = pdd_size

        return None

    def forward(self, input: Tensor) -> Tensor:

        # check input & output size
        kN: int = len(self.kernel_size)
        iN: int = input.dim()
        bN: int = iN - 2 * kN
        oN: int = iN - kN
        input_size: Tuple[int, ...] = tuple(input.shape[bN:])
        batch_size: Tuple[int, ...] = tuple(input.shape[:bN])
        if self.input_size is None:
            fold_input_check(
                input_size=input_size,
                kernel_size=self.kernel_size,
                kernel_position=self.kernel_position,
            )
            self.input_size: Tensor = input_size
            if self.output_size is not None:
                fold_output_check(
                    input_size=input_size,
                    output_size=self.output_size,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride,
                )
            self.build_mask()
        else:
            if not input_size == self.input_size:
                raise ValueError(
                    f"Input tensor does not match previously defined size "
                    f"at kernel correspongding dimensions. "
                    f"Got {input_size}, expected {self.input_size}."
                )

        # permute dimensions
        if self.kernel_position == "last":
            permutation: Tuple[int, ...]
            permutation = tuple([*range(bN), *range(oN, iN), *range(bN, oN)])
            unfold: Tensor = input.permute(permutation)
        # fold
        mask: Tensor = self.mask.expand(size=(*batch_size, *self.mask.shape))
        slided_unfold: Tensor = torch.zeros_like(
            input=mask, dtype=unfold.dtype, device=input.device
        )
        if self.indices is None:
            self.indices: Tuple[Tensor] = mask.flatten().nonzero(as_tuple=True)
        slided_unfold.flatten().scatter_(0, self.indices[0], unfold.flatten())
        reshape: list[int] = [*batch_size, *self.kernel_size, *self.padded_size]
        slided_unfold = slided_unfold.view(reshape)
        dim_reduce: list[int] = list(range(bN, oN))
        fold: Tensor = slided_unfold.sum(dim=dim_reduce)

        # unpad (if specified)
        if any([not p == 0 for p in self.padding]):
            enum: enumerate = enumerate(self.padding)
            aux: list[int] = [slice(p, fold.shape[i + bN] - p) for i, p in enum]
            fold = fold[*[slice(None) for _ in range(bN)], *aux]
        # check correspondence with optional previously specified output size
        if self.output_size is not None:
            cond1: bool = math.prod(fold.shape) == math.prod(self.output_size)
            cond2: bool = math.prod(fold.shape) == math.prod(self.output_size[bN:])
            cond3: bool = math.prod(fold.shape) == math.prod(self.output_size[bN + 1 :])
            if not any([cond1, cond2, cond3]):
                raise ValueError(
                    f"Specified output size {self.output_size} does not match "
                    f"actual folded tensor size {tuple(fold.shape)}."
                )

        return fold


class Unfold(nn.Module):
    """
    A class that implements an n-dimensional unfold operation. Unlike PyTorch's native
    `torch.nn.Unfold`, which only supports 2D unfold operations, this implementation
    generalizes the unfolding process to n-dimensions. The output structure diverges
    from PyTorch's default 2D unfold by returning the output in the shape:

        (*batch_dims, *conv_output_dims, *kernel_dims)

    This means that, instead of flattening the kernel dimensions into one dimension,
    the kernel dimensions are maintained separately in the output, making it more
    intuitive for certain n-dimensional operations.

    This is different from PyTorch's 2D unfold which returns output in the shape:

        (N, C * kernel_height * kernel_width, L)

    where L is the number of sliding windows per input. In this implementation, the
    output retains the kernel dimensions separately rather than collapsing them.

    Additionally, this class allows an optional `input_size` argument to be passed
    during initialization. If `input_size` is provided, it should correspond to the
    non-batched input size (i.e., excluding batch dimensions). If batch dimensions
    are included, they will be ignored. By providing `input_size`, certain
    calculations can be performed during initialization, potentially accelerating the
    `forward` method by avoiding repetitive computation.
    """

    def __init__(
        self,
        kernel_size: _size_any_int,
        dilation: _size_any_int = 1,
        padding: _size_any_int = 0,
        stride: _size_any_int = 1,
        input_size: Optional[_size_any_int] = None,
    ) -> None:
        """
        Args:
            kernel_size (int or tuple of int):
                the size of the sliding window, for each wpatial dimension. If an
                integer is provided, the same value is used for all spatial dimensions.
            dilation (int or tuple of int, optional):
                the spacing between kernel elements along each spatial dimension.
                Default is 1 (no dilation).
            padding (int or tuple of int, optional):
                the amount of padding added to each side of the input along each spatial
                dimension. Default is 0 (no padding).
            stride (int or tuple of int, optional):
                the step size for the sliding window along each spatial dimension.
                Default is 1 (no stride).
            input_size (Optional[int or tuple of int]):
                the size of the input tensor (excluding batch dimensions). If passed,
                this enables pre-calculations at initialization that can speed up the
                `forward` method. If batch dimensions are included, they will be
                ignored. If not provided, these calculations will be performed during
                the `forward` method call.
        """

        super(Unfold, self).__init__()  # This initializes the base Module class

        self.kernel_size: Tuple[int, ...]
        self.dilation: Tuple[int, ...]
        self.padding: Tuple[int, ...]
        self.stride: Tuple[int, ...]
        checked_params: Tuple[Tuple[int, ...], ...] = param_check(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        self.kernel_size, self.dilation, self.padding, self.stride = checked_params

        self.mask: Tensor
        self.unfold_size: Tuple[int, ...]
        self.input_size: Optional[Tuple[int, ...]] = None
        self.input_size = unfold_input_check(input_size, *checked_params)
        if input_size is not None:
            self.build_mask()
        self.indices: Optional[Tuple[Tensor, ...]] = None
        self.expanded_indices = None

        return None

    def build_mask(self) -> None:
        """
        Creates a mask indicating the regions of the input tensor from which elements
        will be selected during the unfolding process. The mask has dimensions:

            (*input_dims, *kernel_element_windows)

        where `kernel_element_windows` represents the projections of each kernel element
        over the input feature space. This is conceptually similar to `kernel windows`,
        but using `kernel element windows` reduces the number of windows per dimension
        from approximately the input dimension size to approximately the kernel
        dimension size.

        The method constructs this mask by considering the padded input size, the
        kernel size, dilation, and stride values, applying these operations to mark the
        regions of the input that correspond to the different windows defined by the
        kernel’s elements.

        Steps:
        1. **Measure dimensionalities**: Determines the number of dimensions in both the
        input and the kernel.
        2. **Compute padded input size**: Adjusts the input size to account for any
        padding applied to the input.
        3. **Define dilated kernel size**: Calculates the effective kernel size after
        applying dilation to the kernel elements.
        4. **Account for mask size**: Initializes the mask with the appropriate size
        based on the padded input dimensions.
        5. **Generate kernel[0,...,0] mask**: Creates a mask for the first kernel
        element by projecting it over the input feature space. This mask is then
        modified to reflect the stride.
        6. **Iterate over kernel dimensions**: For each dimension, the method expands
        the mask and re-indexes the input to account for dilation and stride, repeating
        the process along each kernel dimension.
        7. **Save the final mask**: After iteration, the mask is saved as a class
        attribute, along with the `unfold_size`, which is used during the unfolding
        process.

        The generated mask is used internally by the `Unfold` class to apply the unfold
        operation across n-dimensions, ensuring that only the elements corresponding to
        the kernel windows are selected during unfolding.

        No value is returned, but the mask and unfold size are stored as class
        attributes: (`self.mask` and `self.unfold_size`).
        """

        # meassure dimensionalities (input, kernel, batch)
        kN: int = len(self.kernel_size)
        iN: int = len(self.input_size)
        bN: int = iN - kN
        # compute padded input size
        enum: enumerate = enumerate(self.input_size[bN:])
        pdd_size: Tuple[int, ...] = tuple([sz + 2 * self.padding[i] for i, sz in enum])

        # define dilated kernel size from kernel size and dilation
        dil_size: Callable[[int, int], int] = lambda k, d: k + (k - 1) * (d - 1)
        # kernel[0,...,0] element projection mask over input feature space
        knl0mask: Tensor = torch.full(size=pdd_size, fill_value=False)
        zp: zip = zip(pdd_size, self.kernel_size, self.stride, self.dilation)
        lzp: list[Tuple[int, ...]] = list(zp)
        knl0mask[[slice(i - dil_size(k, d) + 1) for i, k, _, d in lzp]].logical_not_()
        # apply stride to the kernel[0,...,0] mask
        on_stride: Tensor = torch.full_like(input=knl0mask, fill_value=False)
        on_stride[tuple(slice(0, i, s) for i, _, s, _ in lzp)] = True
        knl0mask.logical_and_(on_stride)

        # initialize variables for the loop
        knl_mask: Tensor = knl0mask
        reps: list[int] = []  # to store kernel sizes adjusted by dilation
        ms: Tuple[int, ...] = pdd_size  # to account mask size
        # iterate over kernel dimensions repeating n each along a new dimension
        for i, (si, sk, _, sd) in enumerate(lzp):
            # kernel window adjusted by dilation
            sk: int = dil_size(sk, sd)
            # compute auxiliar step size & next mask size
            kdms: int = len(ms) - kN
            auxstp: list[Union[slice, None]]
            auxstp = [slice(ms[j + kdms]) if j == i else None for j in range(kN)]
            auxstp = [None] * (kdms) + [slice(sk)] + auxstp
            resize: list[int] = list(ms[:kdms]) + [sk] + list(ms[kdms:])
            # compute reindexation indices
            shift: Tensor = torch.arange(start=0, end=(sk * sd), step=sd)[:, None]
            reidx: Tensor
            reidx = (torch.arange(start=0, end=si)[None, :] - shift) % si
            reidx = reidx[auxstp]  # aux step to enssure proper dim assignment
            reidx = reidx.expand(resize)
            # reindex
            knl_mask = knl_mask.unsqueeze(kdms).expand(resize)
            knl_mask = knl_mask.gather(i + kdms + 1, reidx)
            # iteration saves
            ms = tuple(knl_mask.shape)
            reps.append(sk)

        # define class attributes
        self.mask: Tensor = knl_mask
        output_size: list[int] = [(i - d * (k - 1) - 1) // s + 1 for i, k, s, d in lzp]
        self.unfold_size: list[int] = reps + output_size
        self.padded_size: Tuple[int, ...] = pdd_size

        return None

    def forward(self, input: Tensor) -> Tensor:

        # check input size
        kN: int = len(self.kernel_size)
        iN: int = input.dim()
        bN: int = iN - kN
        input_size: Tuple[int, ...] = tuple(input.shape[bN:])
        batch_size: Tuple[int, ...] = tuple(input.shape[:bN])
        if self.input_size is None:
            unfold_input_check(
                input_size=input_size,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
            self.input_size = input_size
            self.build_mask()
        else:
            if not input_size == self.input_size:
                raise ValueError(
                    f"Input tensor does not match previously defined size "
                    f"at kernel correspongding dimensions. "
                    f"Got {input_size}, expected {self.input_size}."
                )

        # apply padding
        aux: list[int] = [x for p in self.padding[::-1] for x in (p, p)]
        input = pad(input=input, pad=aux, value=0)
        # unfold
        broadcast_size: Tuple[int, ...]
        broadcast_size = tuple([*batch_size, *([1] * kN), *self.padded_size])
        mask: Tensor = self.mask.expand(size=(*batch_size, *self.mask.shape))
        input: Tensor = input.view(size=broadcast_size).expand_as(other=mask)
        if self.indices is None:
            self.indices = mask.flatten().nonzero(as_tuple=True)
        flat_unfold: Tensor = input.flatten().take(index=self.indices[0])
        unfold_size: Tuple[int, ...] = tuple([*batch_size, *self.unfold_size])
        reversed_unfold: Tensor = flat_unfold.view(size=unfold_size)
        permutation: Tuple[int, ...]
        permutation = tuple([*range(bN), *range(iN, iN + kN), *range(bN, iN)])
        unfold: Tensor = reversed_unfold.permute(dims=permutation)

        return unfold
