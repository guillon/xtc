#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing import Any
import ctypes

__all__ = [
    "CFunc",
    "_c_ascii_str",
    "_str_list_to_c",
]


# TVM 0.26 FFI ABI
class CTVMFFIAnyValue(ctypes.Union):
    _fields_ = [
        ("v_int64", ctypes.c_int64),
        ("v_float64", ctypes.c_double),
        ("v_ptr", ctypes.c_void_p),
        ("v_c_str", ctypes.c_char_p),
        ("v_uint64", ctypes.c_uint64),
    ]


class CTVMFFIAny(ctypes.Structure):
    _anonymous_ = ("value",)

    _fields_ = [
        ("type_index", ctypes.c_int32),
        ("zero_padding", ctypes.c_uint32),
        ("value", CTVMFFIAnyValue),
    ]


class CTVMFFINDArrayArg(CTVMFFIAny):
    def __init__(self, arg: Any):
        assert arg.__class__.__name__ == "NDArray"
        dl_tensor = arg.handle
        super().__init__(
            type_index=7, zero_padding=0, v_ptr=ctypes.cast(dl_tensor, ctypes.c_void_p)
        )


class CTVMFFIResult(CTVMFFIAny):
    def __init__(self):
        super().__init__()


CTVMFFIPackedFunc = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.POINTER(CTVMFFIAny),
    ctypes.c_int32,
    ctypes.POINTER(CTVMFFIAny),
)


class CFunc:
    _supported_abis = ["bare", "tvm_ffi"]

    def __init__(self, f: Any, abi: str | None = None) -> None:
        self.handle = f
        self.abi = abi
        if self.abi is None:
            if hasattr(self.handle, "packed") and self.handle.packed:
                # TODO: for now infer tvm_ffi abi for packed
                self.abi = "tvm_ffi"
        if self.abi is None:
            self.abi = "bare"
        assert self.abi in self._supported_abis

    def _mangled_arg(self, arg: Any) -> Any:
        if arg.__class__.__name__ == "ndarray":  # Numpy Array
            assert self.abi == "bare"
            return arg.ctypes.data_as(ctypes.c_voidp)
        elif arg.__class__.__name__ == "NDArray":  # TVM NDArray or our NDArray
            if (
                hasattr(arg, "is_on_device") and arg.is_on_device()
            ):  # Device living NDArray
                assert self.abi == "bare", "TODO: device NDArray not supported yet"
            if self.abi == "tvm_ffi":
                assert self.abi == "tvm_ffi"
                return CTVMFFINDArrayArg(arg)
            else:
                assert self.abi == "bare"
                return ctypes.cast(arg.data, ctypes.c_voidp)
        else:
            assert 0, f"Unsupported argument class: {arg.__class__.__name__}"

    def get_args_list(self, args: list[Any]) -> list[Any]:
        return [self._mangled_arg(arg) for arg in args]

    def get_ctypes_args(self, args: list[Any]) -> Any:
        args_list = self.get_args_list(args)
        if self.abi == "tvm_ffi":
            return (CTVMFFIAny * len(args_list))(*args_list)
        else:
            return (ctypes.c_voidp * len(args_list))(*args_list)

    def __call__(self, *args: Any):
        func_addr = ctypes.cast(self.handle, ctypes.c_voidp).value
        assert func_addr is not None
        ctypes_args = self.get_ctypes_args(list(args))
        if self.abi == "tvm_ffi":
            result = CTVMFFIResult()
            ctx = ctypes.c_void_p()
            CTVMFFIPackedFunc(func_addr)(
                ctx,
                ctypes_args,
                len(ctypes_args),
                ctypes.byref(result),
            )
            assert result.v_int64 == 0, f"error calling packed function"
        else:
            func_type = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * len(ctypes_args)))
            func_type(func_addr)(*ctypes_args)


class _c_ascii_str:
    @staticmethod
    def from_param(obj: str | bytes):
        if isinstance(obj, str):
            obj = obj.encode("ascii")
        return ctypes.c_char_p.from_param(obj)


def _str_list_to_c(str_list: list[str]) -> Any:
    return (ctypes.c_char_p * len(str_list))(*[str.encode("utf-8") for str in str_list])
