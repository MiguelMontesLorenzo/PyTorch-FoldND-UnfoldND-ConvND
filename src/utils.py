# Python 3.12

# Standard Library dependencies
from typing import Optional, Sequence, Tuple

# Internal dependencies
from src.internal_types import _size_any_int


def conv_param_check(
    input_channels: int,
    output_channels: int,
    kernel_size: Sequence[int],
    input_size: Optional[Sequence[int]],
    bias: bool,
) -> None:

    if not isinstance(input_channels, int):
        raise TypeError(
            f"Incorrect type for input_channels. "
            f"Got {type(input_channels).__name__}, expected int."
        )

    if not isinstance(output_channels, int):
        raise TypeError(
            f"Incorrect type for output_channels. "
            f"Got {type(output_channels).__name__}, expected int."
        )

    if not isinstance(kernel_size, Sequence):
        raise TypeError(
            f"Incorrect type for kernel_size. "
            f"Got {type(kernel_size).__name__}, expected Tuple[int, ...]."
        )

    if isinstance(kernel_size, Sequence) and not all(
        isinstance(i, int) for i in kernel_size
    ):
        raise TypeError(
            f"Incorrect type for kernel_size. "
            f"Got {type(kernel_size).__name__}, expected Tuple[int, ...]."
        )

    if input_size is not None:
        if not isinstance(input_size, Sequence):
            raise TypeError(
                f"Incorrect type for input_size. "
                f"Got {type(input_size).__name__}, expected Tuple[int, ...]."
            )

        if isinstance(input_size, Sequence) and not all(
            isinstance(i, int) for i in input_size
        ):
            raise TypeError(
                f"Incorrect type for input_size. "
                f"Got {type(input_size).__name__}, expected Tuple[int, ...]."
            )

    if not isinstance(bias, bool):
        raise TypeError(
            f"Incorrect type for bias. "
            f"Got {type(bias).__name__}, expected {bool.__name__}."
        )

    return None


def param_check(
    kernel_size: _size_any_int,
    dilation: _size_any_int = 1,
    padding: _size_any_int = 0,
    stride: _size_any_int = 1,
) -> Tuple[Tuple[int, ...], ...]:

    # Type checks
    args: dict = {
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
    }
    for name, var in args.items():
        if not isinstance(var, (int, Sequence)):
            raise TypeError(
                f"Incorrect type for {name}: "
                f"got {type(var).__name__}, "
                f"expected int or Sequence[int]."
            )
        if isinstance(var, Sequence) and not all(isinstance(i, int) for i in var):
            raise TypeError(
                f"Incorrect type for {name}: "
                f"got {type(var).__name__}, "
                f"expected int or Sequence[int]."
            )

    # Check kernel size
    _kernel_size: Tuple[int, ...]
    if isinstance(kernel_size, int):
        _kernel_size = (kernel_size,)
    else:
        _kernel_size = tuple(kernel_size)

    # Check dilation
    _dilation: Tuple[int, ...]
    if isinstance(dilation, int):
        _dilation = tuple([dilation for _ in enumerate(_kernel_size)])
    else:
        if len(dilation) == 1:
            _dilation = tuple([dilation[0] for _ in enumerate(_kernel_size)])
        elif len(dilation) == len(_kernel_size):
            _dilation = tuple(dilation)
        else:
            raise ValueError(
                f"Dilation must be 1D or match the number of kernel dimensions. "
                f"Got {len(dilation)}D, expected {len(_kernel_size)}."
            )

    # Check padding
    _padding: Tuple[int, ...]
    if isinstance(padding, int):
        _padding = tuple([padding for _ in enumerate(_kernel_size)])
    else:
        if len(padding) == 1:
            _padding = tuple([padding[0] for _ in enumerate(_kernel_size)])
        elif len(padding) == len(_kernel_size):
            _padding = tuple(padding)
        else:
            raise ValueError(
                f"Padding must be 1D or match the number of kernel dimensions. "
                f"Got {len(padding)}D, expected {len(_kernel_size)}."
            )

    # Check stride
    _stride: Tuple[int, ...]
    if isinstance(stride, int):
        _stride = tuple([stride for _ in enumerate(_kernel_size)])
    else:
        if len(stride) == 1:
            _stride = tuple([stride[0] for _ in enumerate(_kernel_size)])
        elif len(stride) == len(_kernel_size):
            _stride = tuple(stride)
        else:
            raise ValueError(
                f"Stride must be 1D or match the number of kernel dimensions. "
                f"Got {len(stride)}D, expected {len(_kernel_size)}."
            )

    checked_param: Tuple[Tuple[int, ...], ...] = (
        _kernel_size,
        _dilation,
        _padding,
        _stride,
    )

    return checked_param


def fold_input_check(
    input_size: Sequence[int],
    kernel_size: Tuple[int, ...],
    kernel_position: str,
) -> Tuple[int, ...]:

    # Check input_size
    _input_size: Tuple[int, ...]
    if isinstance(input_size, int):
        _input_size = (input_size,)
    elif isinstance(input_size, Sequence):
        if not all(isinstance(i, int) for i in input_size):
            raise TypeError(
                f"Incorrect type for input_size: "
                f"got {type(input_size).__name__}, "
                f"expected int or Sequence[int]."
            )
        _input_size = tuple(input_size[len(input_size) - len(kernel_size) :])
    elif input_size is None:
        _input_size = None
    else:
        raise TypeError(
            f"Incorrect type for input_size: "
            f"got {type(input_size).__name__}, "
            f"expected int or Sequence[int]."
        )
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

    if _input_size is not None:
        for i, si in enumerate(_input_size):
            sk: int = kernel_size[i]
            iN: int = len(_input_size)
            kN: int = len(kernel_size)

            if kernel_position == "first":
                if i > iN - 2 * kN and i < iN - kN and not si == sk:
                    raise ValueError(
                        f"Input kernel dimension sizes do not match "
                        f"kernel dimention sizes at position {i} "
                        f"Found input size {si}, expected {sk}."
                    )
            if kernel_position == "last":
                if i > iN - kN and not si == sk:
                    raise ValueError(
                        f"Input kernel dimension sizes do not match "
                        f"kernel dimention sizes at position {i} "
                        f"Found input size {si}, expected {sk}."
                    )

    return _input_size


def fold_output_check(
    input_size: Sequence[int],
    output_size: Sequence[int],
    kernel_size: Tuple[int, ...],
    dilation: Tuple[int, ...],
    padding: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> Tuple[int, ...]:

    # Check input_size
    _output_size: Tuple[int, ...]
    if isinstance(output_size, int):
        _output_size = (output_size,)
    elif isinstance(output_size, Sequence):
        if not all(isinstance(i, int) for i in output_size):
            raise TypeError(
                f"Incorrect type for output_size: "
                f"got {type(output_size).__name__}, "
                f"expected int or Sequence[int]."
            )
        _output_size = tuple(output_size[len(output_size) - len(kernel_size) :])
    elif output_size is None:
        _output_size = None
    else:
        raise TypeError(
            f"Incorrect type for output_size: "
            f"got {type(output_size).__name__}, "
            f"expected int or Sequence[int]."
        )

    if _output_size is not None:
        for i, so in enumerate(_output_size):
            si: int = input_size[i]
            sk: int = kernel_size[i]
            sd: int = dilation[i]
            sp: int = padding[i]
            ss: int = stride[i]
            skd: int = sk + (sk - 1) * (sd - 1)
            if skd > si:
                raise ValueError(
                    f"dilated kernel is bigger than output at dimenion {i}. "
                    f"{skd} > {si}."
                )

            if not (si + 2 * sp - (skd - 1)) % ss == 0:
                raise ValueError(
                    f"size relationship: "
                    f"[(output + 2 padding - kernel - (kernel - 1) (dilation - 1) + 1) "
                    f"% stride == 0] not satisfied at output dimension {i}.",
                )

            if not ss * (si - 1) + sd * (sk - 1) + 1 == so:
                raise ValueError(
                    f"size relationship: "
                    f"[input * (input - 1) - dilation * (kernel - 1) + 1 == output] "
                    f"not satisfied at output dimension {i}.",
                )

    return _output_size


def unfold_input_check(
    input_size: Sequence[int],
    kernel_size: Tuple[int, ...],
    dilation: Tuple[int, ...],
    padding: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> Tuple[int, ...]:

    # Check input_size
    _input_size: Tuple[int, ...]
    if isinstance(input_size, int):
        _input_size = (input_size,)
    elif isinstance(input_size, Sequence):
        if not all(isinstance(i, int) for i in input_size):
            raise TypeError(
                f"Incorrect type for input_size: "
                f"got {type(input_size).__name__}, "
                f"expected int or Sequence[int]."
            )
        _input_size = tuple(input_size[len(input_size) - len(kernel_size) :])
    elif input_size is None:
        _input_size = None
    else:
        raise TypeError(
            f"Incorrect type for input_size: "
            f"got {type(input_size).__name__}, "
            f"expected int or Sequence[int]."
        )

    if _input_size is not None:
        for i, si in enumerate(_input_size):
            sk: int = kernel_size[i]
            sd: int = dilation[i]
            sp: int = padding[i]
            ss: int = stride[i]
            skd: int = sk + (sk - 1) * (sd - 1)
            if skd > si:
                raise ValueError(
                    f"dilated kernel is bigger than input at dimenion {i}. "
                    f"{skd} > {si}."
                )

            if not (si + 2 * sp - (skd - 1)) % ss == 0:
                raise ValueError(
                    f"size relationship: "
                    f"[(input + 2 padding - kernel - (kernel - 1) (dilation - 1) + 1) "
                    f"% stride == 0] not satisfied at input dimension {i}.",
                )

    return _input_size
