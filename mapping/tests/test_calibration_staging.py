import hashlib
from pathlib import Path

import pytest
import zstandard


def _compressed_part(path: Path, payload: bytes) -> None:
    path.write_bytes(zstandard.ZstdCompressor().compress(payload))


def test_staging_uses_a_bounded_process_pool_and_preserves_source_parts(
    monkeypatch, tmp_path
):
    from mapping.pc.go2_mapping_calibration import staging

    source = tmp_path / "segment"
    source.mkdir()
    _compressed_part(source / "segment_0.db3.zstd", b"sqlite-part-zero")
    _compressed_part(source / "segment_1.db3.zstd", b"sqlite-part-one")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.iterdir())
    }
    requested_workers = []

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            requested_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, values):
            return [function(value) for value in values]

    monkeypatch.setattr(staging, "ProcessPoolExecutor", ImmediateExecutor)

    staged = staging.stage_capture_for_diagnostic(
        source, tmp_path / "staging", workers=20
    )

    assert requested_workers == [2]
    assert staged.staged_path.is_dir()
    assert [part.staged_relative_path for part in staged.parts] == [
        "segment_0.db3",
        "segment_1.db3",
    ]
    assert [
        (staged.staged_path / part.staged_relative_path).read_bytes()
        for part in staged.parts
    ] == [b"sqlite-part-zero", b"sqlite-part-one"]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.iterdir())
    } == before
    assert staged.manifest_path.is_file()


def test_staging_does_not_promote_a_partial_directory_when_a_part_is_invalid(tmp_path):
    from mapping.pc.go2_mapping_calibration.staging import DiagnosticStagingError, stage_capture_for_diagnostic

    source = tmp_path / "segment"
    source.mkdir()
    _compressed_part(source / "segment_0.db3.zstd", b"valid")
    (source / "segment_1.db3.zstd").write_bytes(b"not-a-zstd-frame")

    with pytest.raises(DiagnosticStagingError, match="cannot materialize"):
        stage_capture_for_diagnostic(source, tmp_path / "staging", workers=2)

    assert not (tmp_path / "staging" / "segment").exists()
    assert not list((tmp_path / "staging").glob(".segment.partial-*"))


def test_staging_reuses_only_a_complete_stage_matching_its_immutable_source(
    monkeypatch, tmp_path
):
    from mapping.pc.go2_mapping_calibration import staging

    source = tmp_path / "segment"
    source.mkdir()
    _compressed_part(source / "segment_0.db3.zstd", b"first")
    _compressed_part(source / "segment_1.db3.zstd", b"second")
    first = staging.stage_capture_for_diagnostic(source, tmp_path / "staging", workers=2)

    class UnexpectedExecutor:
        def __init__(self, **_kwargs):
            raise AssertionError("a complete matching stage must not be rematerialized")

    monkeypatch.setattr(staging, "ProcessPoolExecutor", UnexpectedExecutor)

    resumed = staging.stage_capture_for_diagnostic(source, tmp_path / "staging", workers=2)

    assert resumed == first
