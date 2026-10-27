"""Read-only, content-addressed ROS 2 bag evidence for calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3
import tempfile
from types import MappingProxyType
from typing import Iterator, Mapping

import zstandard


class BagEvidenceError(ValueError):
    """Raised when a capture cannot be trusted as calibration evidence."""


@dataclass(frozen=True)
class CalibrationTopicMap:
    lidar: str
    imu: str
    lio_pose: str
    color: str
    depth: str
    color_camera_info: str
    depth_camera_info: str
    tf: str
    tf_static: str

    @classmethod
    def defaults(cls) -> "CalibrationTopicMap":
        return cls(
            lidar="/go2/lio/lidar",
            imu="/dog_imu_raw",
            lio_pose="/go2/lio/pose",
            color="/camera/color/image_raw",
            depth="/camera/depth/image_rect_raw",
            color_camera_info="/camera/color/camera_info",
            depth_camera_info="/camera/depth/camera_info",
            tf="/tf",
            tf_static="/tf_static",
        )

    def as_dict(self) -> dict[str, str]:
        return {name: str(value) for name, value in vars(self).items()}


EXPECTED_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "lidar": "sensor_msgs/msg/PointCloud2",
        "imu": "sensor_msgs/msg/Imu",
        "lio_pose": "geometry_msgs/msg/PoseWithCovarianceStamped",
        "color": "sensor_msgs/msg/Image",
        "depth": "sensor_msgs/msg/Image",
        "color_camera_info": "sensor_msgs/msg/CameraInfo",
        "depth_camera_info": "sensor_msgs/msg/CameraInfo",
        "tf": "tf2_msgs/msg/TFMessage",
        "tf_static": "tf2_msgs/msg/TFMessage",
    }
)


@dataclass(frozen=True)
class TopicRecord:
    topic_id: int
    name: str
    type_name: str
    serialization_format: str


@dataclass(frozen=True)
class BagInventory:
    source_path: str
    artifact_sha256: str
    artifact_bytes: int
    duration_s: float
    topics: tuple[TopicRecord, ...]
    message_counts: Mapping[str, int]
    missing_streams: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "duration_s": self.duration_s,
            "topics": [
                {
                    "id": item.topic_id,
                    "name": item.name,
                    "type": item.type_name,
                    "serialization_format": item.serialization_format,
                }
                for item in self.topics
            ],
            "message_counts": dict(self.message_counts),
            "missing_streams": list(self.missing_streams),
        }


@dataclass(frozen=True)
class CapturePart:
    relative_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RuntimeManifest:
    robot_serial: str
    camera_serial: str
    camera_model: str
    camera_firmware: str
    depth_scale_m: float | None
    release_sha256: str
    capture_mode: str
    started_at_utc: str
    finished_at_utc: str
    ros_domain_id: int
    topics: Mapping[str, str]
    robot_identity_source: str
    robot_factory_serial_verified: bool
    camera_identity_source: str


@dataclass(frozen=True)
class CaptureInventory:
    source_path: str
    artifact_sha256: str
    artifact_bytes: int
    part_count: int
    parts: tuple[CapturePart, ...]
    duration_s: float
    topics: tuple[TopicRecord, ...]
    message_counts: Mapping[str, int]
    missing_streams: tuple[str, ...]
    runtime_manifest: RuntimeManifest | None
    runtime_manifest_sha256: str | None
    missing_evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "part_count": self.part_count,
            "parts": [vars(item) for item in self.parts],
            "duration_s": self.duration_s,
            "topics": [vars(item) for item in self.topics],
            "message_counts": dict(self.message_counts),
            "missing_streams": list(self.missing_streams),
            "runtime_manifest": None if self.runtime_manifest is None else {
                **vars(self.runtime_manifest), "topics": dict(self.runtime_manifest.topics)
            },
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "missing_evidence": list(self.missing_evidence),
        }


class CalibrationBag:
    """Materialize an optional zstd bag and expose only read-only queries."""

    def __init__(self, path: str | Path, topics: CalibrationTopicMap | None = None) -> None:
        self.source_path = Path(path).resolve(strict=False)
        name = self.source_path.name.lower()
        if not (name.endswith(".db3") or name.endswith(".db3.zstd")):
            raise BagEvidenceError("calibration evidence must be a .db3 or .db3.zstd ROS 2 bag")
        if not self.source_path.is_file():
            raise BagEvidenceError(f"calibration bag does not exist: {self.source_path}")
        self.topics = topics or CalibrationTopicMap.defaults()
        self.artifact_sha256 = self._sha256(self.source_path)
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._sqlite_path: Path | None = None
        self._inventory: BagInventory | None = None

    def __enter__(self) -> "CalibrationBag":
        if self._sqlite_path is not None:
            raise BagEvidenceError("calibration bag context cannot be entered twice")
        if self.source_path.name.lower().endswith(".zstd"):
            self._temporary = tempfile.TemporaryDirectory(prefix="go2-calibration-bag-")
            self._sqlite_path = Path(self._temporary.name) / "capture.db3"
            try:
                with self.source_path.open("rb") as source, self._sqlite_path.open("xb") as destination:
                    zstandard.ZstdDecompressor().copy_stream(source, destination)
            except (OSError, zstandard.ZstdError) as exc:
                self.close()
                raise BagEvidenceError(f"cannot decompress calibration bag: {self.source_path}") from exc
        else:
            self._sqlite_path = self.source_path
        self._validate_sqlite()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @property
    def sqlite_path(self) -> Path:
        if self._sqlite_path is None:
            raise BagEvidenceError("calibration bag must be used as a context manager")
        return self._sqlite_path

    def close(self) -> None:
        temporary = self._temporary
        self._temporary = None
        self._sqlite_path = None
        if temporary is not None:
            temporary.cleanup()

    def inventory(self) -> BagInventory:
        if self._inventory is not None:
            return self._inventory
        with closing(self._connect()) as database:
            columns = {str(row[1]) for row in database.execute("PRAGMA table_info(topics)")}
            serialization = "serialization_format" if "serialization_format" in columns else "'cdr'"
            topic_rows = tuple(
                TopicRecord(int(row[0]), str(row[1]), str(row[2]), str(row[3]))
                for row in database.execute(
                    f"SELECT id, name, type, {serialization} FROM topics ORDER BY id"
                )
            )
            counts_by_id = {
                int(row[0]): int(row[1])
                for row in database.execute(
                    "SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"
                )
            }
            bounds = database.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
        observed = {item.name: item for item in topic_rows}
        message_counts: dict[str, int] = {}
        missing: list[str] = []
        for stream, topic_name in self.topics.as_dict().items():
            topic = observed.get(topic_name)
            count = 0 if topic is None else counts_by_id.get(topic.topic_id, 0)
            message_counts[stream] = count
            if (
                topic is None
                or topic.type_name != EXPECTED_TYPES[stream]
                or topic.serialization_format not in {"", "cdr"}
                or count <= 0
            ):
                missing.append(stream)
        minimum, maximum = bounds if bounds is not None else (None, None)
        duration_s = 0.0 if minimum is None or maximum is None else max(0.0, (maximum - minimum) / 1e9)
        self._inventory = BagInventory(
            source_path=str(self.source_path),
            artifact_sha256=self.artifact_sha256,
            artifact_bytes=self.source_path.stat().st_size,
            duration_s=float(duration_s),
            topics=topic_rows,
            message_counts=MappingProxyType(message_counts),
            missing_streams=tuple(missing),
        )
        return self._inventory

    def require_complete_inventory(self) -> BagInventory:
        inventory = self.inventory()
        if inventory.missing_streams:
            raise BagEvidenceError(
                "missing or invalid calibration streams: " + ", ".join(inventory.missing_streams)
            )
        return inventory

    def iter_messages(self, stream: str) -> Iterator[tuple[int, bytes]]:
        if stream not in EXPECTED_TYPES:
            raise BagEvidenceError(f"unknown calibration stream: {stream}")
        topic_name = self.topics.as_dict()[stream]
        with closing(self._connect()) as database:
            query = (
                "SELECT messages.timestamp, messages.data FROM messages "
                "JOIN topics ON topics.id = messages.topic_id "
                "WHERE topics.name = ? AND topics.type = ? ORDER BY messages.timestamp, messages.id"
            )
            for timestamp, payload in database.execute(query, (topic_name, EXPECTED_TYPES[stream])):
                yield int(timestamp), bytes(payload)

    def _validate_sqlite(self) -> None:
        try:
            with closing(self._connect()) as database:
                tables = {str(row[0]) for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"topics", "messages"}.issubset(tables):
                    raise BagEvidenceError("ROS 2 bag is missing topics/messages tables")
                result = database.execute("PRAGMA quick_check").fetchone()
                if result is None or result[0] != "ok":
                    raise BagEvidenceError("ROS 2 bag failed SQLite integrity check")
        except sqlite3.Error as exc:
            raise BagEvidenceError(f"cannot open calibration bag read-only: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        uri = self.sqlite_path.as_uri() + "?mode=ro&immutable=1"
        database = sqlite3.connect(uri, uri=True)
        database.execute("PRAGMA query_only=ON")
        return database

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class CalibrationCapture:
    """One logical capture backed by one bag or a directory of immutable parts."""

    def __init__(
        self,
        path: str | Path,
        topics: CalibrationTopicMap | None = None,
        *,
        materialization_workers: int = 1,
    ) -> None:
        self.source_path = Path(path).resolve(strict=False)
        self.topics = topics or CalibrationTopicMap.defaults()
        if int(materialization_workers) <= 0:
            raise ValueError("materialization_workers must be positive")
        self.materialization_workers = int(materialization_workers)
        if self.source_path.is_dir():
            candidates = sorted(
                path
                for path in self.source_path.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and (path.name.lower().endswith(".db3") or path.name.lower().endswith(".db3.zstd"))
            )
            if not candidates:
                raise BagEvidenceError(f"capture directory has no published db3 parts: {self.source_path}")
            self._paths = tuple(candidates)
        elif self.source_path.is_file():
            self._paths = (self.source_path,)
        else:
            raise BagEvidenceError(f"calibration capture does not exist: {self.source_path}")
        self._bags: list[CalibrationBag] = []
        self._inventory: CaptureInventory | None = None

    def _open_bag(self, path: Path) -> CalibrationBag:
        return CalibrationBag(path, self.topics).__enter__()

    def _runtime_manifest(self) -> tuple[RuntimeManifest | None, str | None]:
        if not self.source_path.is_dir():
            return None, None
        sidecar = self.source_path / "capture.runtime.json"
        if not sidecar.is_file():
            return None, None
        import json
        try:
            raw = sidecar.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BagEvidenceError(f"runtime manifest cannot be decoded: {exc}") from exc
        required = {"schema_version", "robot", "camera", "release_sha256", "capture_mode", "started_at_utc", "finished_at_utc", "ros_domain_id", "topics"}
        if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 1:
            raise BagEvidenceError("runtime manifest has unknown/missing fields or schema")
        robot, camera, topics = payload["robot"], payload["camera"], payload["topics"]
        if not isinstance(robot, dict) or set(robot) != {"model", "serial", "identity_source", "factory_serial_verified"} or robot.get("model") != "go2_edu":
            raise BagEvidenceError("runtime manifest robot identity is invalid")
        if not isinstance(robot["factory_serial_verified"], bool):
            raise BagEvidenceError("runtime manifest robot serial verification flag is invalid")
        if not isinstance(camera, dict) or set(camera) not in ({"model", "serial", "firmware", "identity_source"}, {"model", "serial", "firmware", "identity_source", "depth_scale_m"}):
            raise BagEvidenceError("runtime manifest camera identity is invalid")
        if not isinstance(topics, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in topics.items()):
            raise BagEvidenceError("runtime manifest topic contract is invalid")
        if set(topics.items()) != {
            (topic.name, topic.type_name)
            for inventory in [bag.inventory() for bag in self._bags]
            for topic in inventory.topics
        }:
            raise BagEvidenceError("runtime manifest topic contract does not match recorded topics")
        release = str(payload["release_sha256"])
        if len(release) != 64 or any(value not in "0123456789abcdefABCDEF" for value in release):
            raise BagEvidenceError("runtime manifest release_sha256 is invalid")
        if not all(str(value).strip() for value in (robot["serial"], robot["identity_source"], camera["model"], camera["serial"], camera["firmware"], camera["identity_source"], payload["capture_mode"], payload["started_at_utc"], payload["finished_at_utc"])):
            raise BagEvidenceError("runtime manifest identity/time fields are empty")
        if not isinstance(payload["ros_domain_id"], int) or not 0 <= payload["ros_domain_id"] <= 232:
            raise BagEvidenceError("runtime manifest ros_domain_id is invalid")
        depth_scale = camera.get("depth_scale_m")
        if depth_scale is not None:
            try:
                depth_scale = float(depth_scale)
            except (TypeError, ValueError) as exc:
                raise BagEvidenceError("runtime manifest depth scale is invalid") from exc
            if not 0.00001 <= depth_scale <= 0.1:
                raise BagEvidenceError("runtime manifest depth scale is outside physical bounds")
        return RuntimeManifest(
            robot_serial=str(robot["serial"]), camera_serial=str(camera["serial"]), camera_model=str(camera["model"]), camera_firmware=str(camera["firmware"]),
            depth_scale_m=depth_scale,
            release_sha256=release.lower(), capture_mode=str(payload["capture_mode"]), started_at_utc=str(payload["started_at_utc"]), finished_at_utc=str(payload["finished_at_utc"]),
            ros_domain_id=int(payload["ros_domain_id"]), topics=MappingProxyType(dict(topics)),
            robot_identity_source=str(robot["identity_source"]),
            robot_factory_serial_verified=bool(robot["factory_serial_verified"]),
            camera_identity_source=str(camera["identity_source"]),
        ), hashlib.sha256(raw).hexdigest()

    def __enter__(self) -> "CalibrationCapture":
        opened: list[CalibrationBag] = []
        try:
            worker_count = min(self.materialization_workers, len(self._paths))
            if worker_count == 1:
                opened = [self._open_bag(path) for path in self._paths]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    opened = list(executor.map(self._open_bag, self._paths))
            self._bags.extend(opened)
        except Exception:
            for bag in reversed(opened):
                bag.close()
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        for bag in reversed(self._bags):
            bag.close()
        self._bags.clear()

    def inventory(self) -> CaptureInventory:
        if not self._bags:
            raise BagEvidenceError("calibration capture must be used as a context manager")
        if self._inventory is not None:
            return self._inventory
        inventories = [bag.inventory() for bag in self._bags]
        runtime_manifest, runtime_manifest_sha256 = self._runtime_manifest()
        parts = tuple(
            CapturePart(path.relative_to(self.source_path).as_posix() if self.source_path.is_dir() else path.name, inventory.artifact_sha256, inventory.artifact_bytes)
            for path, inventory in zip(self._paths, inventories)
        )
        identity = [
            {"relative_path": item.relative_path, "sha256": item.sha256, "bytes": item.bytes}
            for item in parts
        ]
        if runtime_manifest_sha256 is not None:
            identity.append({"relative_path": "capture.runtime.json", "sha256": runtime_manifest_sha256, "bytes": (self.source_path / "capture.runtime.json").stat().st_size})
        digest = hashlib.sha256(
            json_bytes(identity)
        ).hexdigest()
        message_counts = {
            stream: sum(item.message_counts.get(stream, 0) for item in inventories)
            for stream in EXPECTED_TYPES
        }
        missing = tuple(
            stream
            for stream, expected_type in EXPECTED_TYPES.items()
            if message_counts[stream] <= 0
            or any(
                topic.name == self.topics.as_dict()[stream] and topic.type_name != expected_type
                for inventory in inventories
                for topic in inventory.topics
            )
        )
        topic_by_identity = {
            (topic.name, topic.type_name, topic.serialization_format): topic
            for inventory in inventories for topic in inventory.topics
        }
        self._inventory = CaptureInventory(
            source_path=str(self.source_path),
            artifact_sha256=digest,
            artifact_bytes=sum(item.bytes for item in parts),
            part_count=len(parts),
            parts=parts,
            duration_s=sum(item.duration_s for item in inventories),
            topics=tuple(sorted(topic_by_identity.values(), key=lambda item: (item.name, item.type_name))),
            message_counts=MappingProxyType(message_counts),
            missing_streams=missing,
            runtime_manifest=runtime_manifest,
            runtime_manifest_sha256=runtime_manifest_sha256,
            missing_evidence=() if runtime_manifest is not None else ("runtime_manifest",),
        )
        return self._inventory

    def require_complete_inventory(self) -> CaptureInventory:
        inventory = self.inventory()
        if inventory.missing_streams:
            raise BagEvidenceError("missing or invalid calibration streams: " + ", ".join(inventory.missing_streams))
        return inventory

    def iter_messages(self, stream: str) -> Iterator[tuple[int, bytes]]:
        messages = []
        for part_index, bag in enumerate(self._bags):
            messages.extend((timestamp, part_index, index, payload) for index, (timestamp, payload) in enumerate(bag.iter_messages(stream)))
        for timestamp, _part, _index, payload in sorted(messages, key=lambda item: item[:3]):
            yield timestamp, payload


def json_bytes(value: object) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
