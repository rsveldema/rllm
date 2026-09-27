import json
import struct

import pytest

import export_identifier_embeddings as exporter


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, rows in tensors.items():
        begin = len(payload)
        for row in rows:
            payload.extend(struct.pack(f"<{len(row)}e", *row))
        header[name] = {
            "dtype": "F16",
            "shape": [len(rows), len(rows[0])],
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_exports_projector_vectors_and_metadata(tmp_path):
    checkpoint = tmp_path / "model.st"
    vectors = tmp_path / "vectors.tsv"
    metadata = tmp_path / "metadata.tsv"
    write_safetensors(checkpoint, {exporter.TENSOR_NAME: [[1, 2], [3, 4], [5, 6]]})

    assert exporter.export_projector_files(checkpoint, vectors, metadata) == (3, 2)
    assert vectors.read_text() == "1\t2\n3\t4\n5\t6\n"
    metadata_rows = metadata.read_text().splitlines()
    assert metadata_rows[0] == "engram_examples\tbucket\tl2_norm"
    assert [row.split("\t")[1:] for row in metadata_rows[1:]] == [
        ["0", "2.23606798"],
        ["1", "5"],
        ["2", "7.81024968"],
    ]
    assert all(row.split("\t")[0] for row in metadata_rows[1:])


def test_representative_engrams_match_their_bucket():
    examples = exporter.representative_engrams(512)

    assert len(examples) == 512
    assert all(examples)
    for bucket, labels in enumerate(examples):
        assert all(exporter.engram_hash(label, 512) == bucket for label in labels)


def test_missing_identifier_embeddings_reports_available_tensor(tmp_path):
    checkpoint = tmp_path / "old-model.st"
    write_safetensors(checkpoint, {"input_layer.embeddings": [[0, 0], [0, 0]]})

    with pytest.raises(KeyError, match="input_layer.embeddings"):
        exporter.tensor_descriptor(checkpoint)
