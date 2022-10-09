# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: npproto/ndarray.proto
# plugin: python-betterproto
from dataclasses import dataclass
from typing import List

import betterproto


@dataclass(eq=False, repr=False)
class Ndarray(betterproto.Message):
    """
    Represents a NumPy array of arbitrary shape or dtype. Note that the array
    must support the buffer protocol.
    """

    data: bytes = betterproto.bytes_field(1)
    dtype: str = betterproto.string_field(2)
    shape: List[int] = betterproto.int64_field(3)
    strides: List[int] = betterproto.int64_field(4)
    order: str = betterproto.string_field(5)
