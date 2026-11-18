"""Checkpoints as .safetensors (the standard weights format: 8 byte header length, a
JSON header, the raw little-endian arrays; readable from torch / jax / numpy / JS /
Hugging Face). Written and read with numpy only (no safetensors package needed).

The header's "__metadata__" holds one key, "webgym": JSON with everything that is not
an array (env, algo, iteration, history, ...).
"""

import json
import struct

import numpy as np

DTYPES = {"F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32}
NAMES = {np.dtype(v): k for k, v in DTYPES.items()}


def dumps(tensors, metadata):
    """bytes of a .safetensors file with `tensors` ({name: array}) and `metadata`
    (JSON-able dict)"""
    header, chunks, offset = {}, [], 0
    for name, array in tensors.items():
        array = np.asarray(array, order="C")  # (not ascontiguousarray: keeps 0-d arrays 0-d)
        if array.dtype not in NAMES:
            array = array.astype(np.float64)
        data = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": NAMES[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        chunks.append(data)
        offset += len(data)
    header["__metadata__"] = {"webgym": json.dumps(metadata)}
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # (alignment of the data, as the format recommends)
    return struct.pack("<Q", len(head)) + head + b"".join(chunks)


def loads(data):
    """(tensors, metadata) of .safetensors bytes"""
    (length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + length])
    metadata = json.loads(header.pop("__metadata__", {}).get("webgym", "{}"))
    body = memoryview(data)[8 + length :]
    tensors = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        array = np.frombuffer(body[start:end], dtype=np.dtype(DTYPES[info["dtype"]]).newbyteorder("<"))
        tensors[name] = array.reshape(info["shape"]).astype(DTYPES[info["dtype"]])
    return tensors, metadata
