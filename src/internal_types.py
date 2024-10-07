# Python 3.12

# Standard Library dependencies
from typing import Sequence, TypeVar, Union


T: TypeVar = TypeVar("T")
type _scalar_or_tuple_any_t[T] = Union[T, Sequence[T]]
type _size_any_int = _scalar_or_tuple_any_t[int]
