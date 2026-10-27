import sqlite3
import hashlib

import zstandard

from mapping.pc.go2_mapping_live import progressive_ingest
from mapping.pc.go2_mapping_live.progressive_ingest import VerifiedArtifactIngestor
from mapping.pc.artifact_lifecycle import ArtifactLifecycleStore


class RecordingMapper:
    def __init__(self):
        self.paths = []

    def ingest_sqlite_bag(self, path):
        self.paths.append(path)
        return {"duplicate": False, "integrated_frames": 2, "integrated_points": 400}


class SqliteRecordingMapper(RecordingMapper):
    def ingest_sqlite_bag(self, path):
        with sqlite3.connect(path) as database:
            self.values = [row[0] for row in database.execute("SELECT value FROM samples")]
        return super().ingest_sqlite_bag(path)


class SidecarWritingMapper(SqliteRecordingMapper):
    def ingest_sqlite_bag(self, path):
        for suffix in ("-shm", "-wal", "-journal"):
            path.with_name(path.name + suffix).write_bytes(b"temporary")
        return super().ingest_sqlite_bag(path)


class ProvenanceRecordingMapper(SqliteRecordingMapper):
    def ingest_sqlite_bag(self, path, *, artifact_sha256, source_artifact_path):
        self.artifact_sha256 = artifact_sha256
        self.source_artifact_path = source_artifact_path
        return super().ingest_sqlite_bag(path)


class RefreshRecordingMapper(RecordingMapper):
    def __init__(self):
        super().__init__()
        self.refreshed_paths = []

    def refresh_derived_artifact(self, path, *, artifact_sha256, source_artifact_path):
        self.refreshed_paths.append(source_artifact_path)
        return True


class DeferredLifecycleStore:
    def __init__(self):
        self.enabled = False
        self.calls = []

    def mark_reconstructed(self, path, *, result):
        self.calls.append(path)
        if not self.enabled:
            raise LookupError(path)
        return {"release_authorized": True}


def test_verified_artifact_ingestor_reads_only_atomically_published_db3_files_once(tmp_path):
    root = tmp_path / "inbox"
    published = root / "lidar" / "segment-000.db3"
    partial = root / ".progressive" / "session" / "artifact" / "00000000.part"
    published.parent.mkdir(parents=True)
    partial.parent.mkdir(parents=True)
    published.write_bytes(b"verified")
    partial.write_bytes(b"not publishable")
    mapper = RecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    first = ingestor.ingest_once()
    second = ingestor.ingest_once()

    assert first["ingested_artifacts"] == 1
    assert second["ingested_artifacts"] == 0
    assert mapper.paths == [published]
    assert ingestor.snapshot()["integrated_points"] == 400


def test_verified_artifact_ingestor_bounds_new_artifacts_per_poll(tmp_path):
    root = tmp_path / "inbox"
    published = [root / "lidar" / f"segment-{index:03d}.db3" for index in range(3)]
    for path in published:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"verified-{path.stem}".encode())
    mapper = RecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    first = ingestor.ingest_once()
    second = ingestor.ingest_once()
    third = ingestor.ingest_once()

    assert [first["ingested_artifacts"], second["ingested_artifacts"], third["ingested_artifacts"]] == [1, 1, 1]
    assert mapper.paths == published


def test_verified_artifact_ingestor_refreshes_one_derived_artifact_only_after_slow_cadence(
    tmp_path,
    monkeypatch,
):
    now = [100.0]
    monkeypatch.setattr(progressive_ingest, "_monotonic", lambda: now[0], raising=False)
    root = tmp_path / "inbox"
    published = root / "lidar" / "segment-000.db3"
    published.parent.mkdir(parents=True)
    published.write_bytes(b"verified")
    mapper = RefreshRecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    assert ingestor.ingest_once()["ingested_artifacts"] == 1
    for _ in range(3):
        assert ingestor.ingest_once()["ingested_artifacts"] == 0
    assert mapper.refreshed_paths == []

    now[0] += 30.0
    ingestor.ingest_once()
    ingestor.ingest_once()

    assert mapper.refreshed_paths == [published]


def test_verified_artifact_ingestor_bounds_late_lifecycle_reconciliation_per_poll(tmp_path):
    root = tmp_path / "inbox"
    published = [root / "lidar" / f"segment-{index:03d}.db3" for index in range(6)]
    for path in published:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"verified-{path.stem}".encode())
    lifecycle = DeferredLifecycleStore()
    ingestor = VerifiedArtifactIngestor(root, RecordingMapper(), lifecycle_store=lifecycle)

    for _ in published:
        assert ingestor.ingest_once()["ingested_artifacts"] == 1
    lifecycle.enabled = True
    lifecycle.calls.clear()

    ingestor.ingest_once()
    first_batch = list(lifecycle.calls)
    lifecycle.calls.clear()
    ingestor.ingest_once()

    assert len(first_batch) == 4
    assert len(lifecycle.calls) == 2
    assert set(first_batch + lifecycle.calls) == set(published)


def test_verified_artifact_ingestor_skips_explicitly_deprecated_segment(tmp_path):
    root = tmp_path / "inbox"
    segment = root / "lidar" / "segment-retired"
    published = segment / "segment-retired_0.db3"
    segment.mkdir(parents=True)
    published.write_bytes(b"must-not-be-ingested")
    (segment / "DEPRECATED.json").write_text(
        '{"status":"DEPRECATED","ingest_authorized":false}\n', encoding="utf-8"
    )

    mapper = RecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    assert ingestor.ingest_once() == {"ingested_artifacts": 0, "errors": 0}
    assert mapper.paths == []


def test_verified_artifact_ingestor_decompresses_published_zstd_sqlite_bag_to_a_temporary_file(tmp_path):
    root = tmp_path / "inbox"
    source_bag = tmp_path / "source.db3"
    published = root / "lidar" / "segment-000.db3.zstd"
    with sqlite3.connect(source_bag) as database:
        database.execute("CREATE TABLE samples (value INTEGER)")
        database.execute("INSERT INTO samples VALUES (42)")
    published.parent.mkdir(parents=True)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)

    mapper = SqliteRecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    result = ingestor.ingest_once()

    assert result == {"ingested_artifacts": 1, "errors": 0}
    assert mapper.values == [42]
    assert len(mapper.paths) == 1
    assert mapper.paths[0].suffix == ".db3"
    assert not mapper.paths[0].exists()
    assert published.exists()


def test_verified_artifact_ingestor_preserves_published_compressed_artifact_provenance(tmp_path):
    root = tmp_path / "inbox"
    source_bag = tmp_path / "source.db3"
    published = root / "lidar" / "segment-000.db3.zstd"
    with sqlite3.connect(source_bag) as database:
        database.execute("CREATE TABLE samples (value INTEGER)")
        database.execute("INSERT INTO samples VALUES (42)")
    published.parent.mkdir(parents=True)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)
    published_digest = hashlib.sha256(published.read_bytes()).hexdigest()
    uncompressed_digest = hashlib.sha256(source_bag.read_bytes()).hexdigest()
    assert published_digest != uncompressed_digest

    mapper = ProvenanceRecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper)

    assert ingestor.ingest_once() == {"ingested_artifacts": 1, "errors": 0}
    assert mapper.values == [42]
    assert mapper.artifact_sha256 == published_digest
    assert mapper.source_artifact_path == published
    assert mapper.paths[0] != published
    assert not mapper.paths[0].exists()


def test_successful_dense_ingest_authorizes_robot_release_of_that_exact_artifact(tmp_path):
    root = tmp_path / "inbox"
    source_bag = tmp_path / "source.db3"
    published = root / "lidar" / "segment-000.db3.zstd"
    with sqlite3.connect(source_bag) as database:
        database.execute("CREATE TABLE samples (value INTEGER)")
        database.execute("INSERT INTO samples VALUES (42)")
    published.parent.mkdir(parents=True)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    lifecycle = ArtifactLifecycleStore(root)
    lifecycle.record_verified("segment-a", digest, published.relative_to(root).as_posix(), published.stat().st_size)

    ingestor = VerifiedArtifactIngestor(root, SqliteRecordingMapper(), lifecycle_store=lifecycle)
    assert ingestor.ingest_once()["ingested_artifacts"] == 1

    status = lifecycle.status("segment-a", digest)
    assert status["release_authorized"] is True
    assert status["integrated_frames"] == 2
    assert status["integrated_points"] == 400
    assert ingestor.snapshot()["release_authorized_artifacts"] == 1


def test_late_lifecycle_record_is_reconciled_without_reingesting_artifact(tmp_path):
    root = tmp_path / "inbox"
    published = root / "lidar" / "segment-000.db3"
    published.parent.mkdir(parents=True)
    published.write_bytes(b"verified")
    digest = hashlib.sha256(published.read_bytes()).hexdigest()
    lifecycle = ArtifactLifecycleStore(root)
    mapper = RecordingMapper()
    ingestor = VerifiedArtifactIngestor(root, mapper, lifecycle_store=lifecycle)

    assert ingestor.ingest_once() == {"ingested_artifacts": 1, "errors": 0}
    assert lifecycle.status("segment-a", digest)["state"] == "UNKNOWN"

    lifecycle.record_verified(
        "segment-a",
        digest,
        published.relative_to(root).as_posix(),
        published.stat().st_size,
    )

    assert ingestor.ingest_once() == {"ingested_artifacts": 0, "errors": 0}
    assert ingestor.ingest_once() == {"ingested_artifacts": 0, "errors": 0}
    assert mapper.paths == [published]
    assert lifecycle.status("segment-a", digest)["state"] == "PC_RECONSTRUCTION_INGESTED"
    assert lifecycle.status("segment-a", digest)["release_authorized"] is True
    assert ingestor.snapshot()["integrated_artifacts"] == 1
    assert ingestor.snapshot()["release_authorized_artifacts"] == 1


def test_verified_artifact_ingestor_removes_zstd_sqlite_sidecars_after_mapping(tmp_path):
    root = tmp_path / "inbox"
    source_bag = tmp_path / "source.db3"
    published = root / "lidar" / "segment-000.db3.zstd"
    with sqlite3.connect(source_bag) as database:
        database.execute("CREATE TABLE samples (value INTEGER)")
        database.execute("INSERT INTO samples VALUES (42)")
    published.parent.mkdir(parents=True)
    with source_bag.open("rb") as source, published.open("wb") as destination:
        zstandard.ZstdCompressor().copy_stream(source, destination)

    ingestor = VerifiedArtifactIngestor(root, SidecarWritingMapper())

    assert ingestor.ingest_once() == {"ingested_artifacts": 1, "errors": 0}
    assert list((root / ".dense-tmp").iterdir()) == []
