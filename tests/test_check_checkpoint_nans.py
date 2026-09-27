import json
import struct

import check_checkpoint_nans as checker


def write_safetensors(path, tensors):
    header = {}
    data = bytearray()
    for name, (dtype, shape, values) in tensors.items():
        formats = {"F16": "e", "F32": "f", "F64": "d"}
        encoded = struct.pack(f"<{len(values)}{formats[dtype]}", *values)
        start = len(data)
        data.extend(encoded)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    encoded_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + data)


def test_clean_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "clean.st"
    write_safetensors(checkpoint, {
        "weights": ("F32", [3], [1.0, 2.0, 3.0]),
        "half_weights": ("F16", [1], [4.0]),
    })

    assert checker.main([str(checkpoint)]) == 0
    assert "OK: no NaNs in 4 floating-point values across 2 tensor(s)" in capsys.readouterr().out


def test_reports_nan_tensor_and_flat_index(tmp_path, capsys):
    checkpoint = tmp_path / "nan.st"
    write_safetensors(checkpoint, {
        "good": ("F64", [1], [1.0]),
        "bad": ("F32", [2, 2], [0.0, float("nan"), 2.0, float("nan")]),
    })

    assert checker.main([str(checkpoint)]) == 1
    output = capsys.readouterr().out
    assert "FOUND 2 NaN value(s) in 1 tensor(s)" in output
    assert "bad: 2 NaN(s), first flat index 1, shape [2, 2]" in output


def test_rejects_malformed_checkpoint(tmp_path, capsys):
    checkpoint = tmp_path / "broken.st"
    checkpoint.write_bytes(b"broken")

    assert checker.main([str(checkpoint)]) == 2
    assert "too short" in capsys.readouterr().err
