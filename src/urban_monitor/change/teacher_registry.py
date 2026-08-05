from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Callable


class TeacherStatus(str, Enum):
    READY = "ready"
    MISSING = "missing"
    DISABLED = "disabled"
    CHECKPOINT_MISSING = "checkpoint_missing"
    MAPPING_PENDING = "mapping_pending"
    DEPENDENCY_MISSING = "dependency_missing"
    PARTIAL_READY = "partial_ready"
    EXTERNAL_READY = "external_ready"


@dataclass(frozen=True)
class TeacherMetadata:
    name: str
    role: str
    status: TeacherStatus
    dataset_root: str | None = None
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    outputs: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["outputs"] = list(self.outputs)
        return payload


@dataclass
class TeacherOutputs:
    teacher: str
    values: dict[str, Any]
    status: TeacherStatus = TeacherStatus.READY
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherPlugin:
    def __init__(
        self,
        metadata: TeacherMetadata,
        predictor: Callable[..., dict[str, Any]] | None = None,
    ):
        self._metadata = metadata
        self._predictor = predictor

    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def available(self) -> bool:
        # An external teacher is deliberately not imported into this runtime.
        # It is usable only through its declared raster-output contract.
        return self._metadata.status in {TeacherStatus.READY, TeacherStatus.PARTIAL_READY}

    def available_outputs(self) -> tuple[str, ...]:
        return self._metadata.outputs if self.available() else ()

    def preflight(self) -> dict[str, Any]:
        return self._metadata.to_dict()

    def predict(self, *args, **kwargs) -> TeacherOutputs:
        if not self.available():
            raise RuntimeError(f"{self._metadata.name} teacher is {self._metadata.status.value}")
        if self._predictor is None:
            raise RuntimeError(f"{self._metadata.name} predictor is not configured")
        return TeacherOutputs(self._metadata.name, self._predictor(*args, **kwargs))


class TeacherRegistry:
    def __init__(self, teachers: list[TeacherPlugin] | None = None):
        self._teachers = {teacher.metadata().name: teacher for teacher in teachers or []}

    def register(self, teacher: TeacherPlugin) -> None:
        name = teacher.metadata().name
        if name in self._teachers:
            raise ValueError(f"Teacher already registered: {name}")
        self._teachers[name] = teacher

    def get(self, name: str) -> TeacherPlugin:
        return self._teachers[name]

    def statuses(self) -> dict[str, dict[str, Any]]:
        return {name: teacher.preflight() for name, teacher in self._teachers.items()}

    def ready(self) -> list[TeacherPlugin]:
        return [teacher for teacher in self._teachers.values() if teacher.available()]

    def __iter__(self):
        return iter(self._teachers.values())


ROLES = {
    "dsifn": ("any_change", ("any_change_probability", "any_change_mask")),
    "s2looking": (
        "building_direction",
        ("building_change_probability", "new_building_probability", "demolition_probability"),
    ),
    "second": ("semantic_change", ("semantic_t1", "semantic_t2", "semantic_transition")),
    "rsbuilding": (
        "building_geometry",
        (
            "building_t1_probability",
            "building_t2_probability",
            "building_t1_mask",
            "building_t2_mask",
            "building_boundary",
        ),
    ),
}


def _path_status(enabled: bool, root: Path | None, checkpoint: Path | None) -> TeacherStatus:
    if not enabled:
        return TeacherStatus.DISABLED
    if root is not None and (not root.is_dir() or not any(root.iterdir())):
        return TeacherStatus.MISSING
    if checkpoint is not None and not checkpoint.is_file():
        return TeacherStatus.CHECKPOINT_MISSING
    return TeacherStatus.READY


def build_teacher_registry(config: dict) -> TeacherRegistry:
    values = config.get("teachers", {})
    datasets = config.get("datasets", {})
    teachers: list[TeacherPlugin] = []
    for name, (role, outputs) in ROLES.items():
        item = {**datasets.get(name, {}), **values.get(name, {})}
        enabled = bool(item.get("enabled", True))
        root = Path(item["root"]) if item.get("root") else None
        checkpoint = Path(item["checkpoint"]) if item.get("checkpoint") else None
        # RSBuilding is an isolated external integration.  Its repository and
        # selected weight are sufficient for an honest EXTERNAL_READY state;
        # precomputed scene rasters are optional at training time.
        if name == "rsbuilding":
            repo = Path(item["repo"]) if item.get("repo") else root
            status = _path_status(enabled, repo, checkpoint)
            if status == TeacherStatus.READY and item.get("external_inference", False):
                status = TeacherStatus.EXTERNAL_READY
            root = repo
        else:
            status = _path_status(enabled, root, checkpoint)
        available_outputs = outputs
        details: dict[str, Any] = {}
        if name == "second" and status == TeacherStatus.READY and not item.get("mapping_verified", False):
            # The raw semantic masks are usable with the configured ignore
            # value, but their building mapping is intentionally unverified.
            status = TeacherStatus.PARTIAL_READY
            details.update({"semantic_supervision": "available", "building_mapping": "pending", "directional_supervision": "disabled"})
        if name == "rsbuilding" and status == TeacherStatus.EXTERNAL_READY:
            output_root = Path(item.get("output_root", root / "outputs" if root else ""))
            details.update({
                "external_inference": True,
                "repository": str(root) if root else None,
                "output_root": str(output_root),
                "scene_exports_present": output_root.is_dir(),
                "dependency_isolated": True,
            })
        checkpoint_hash = sha256_file(checkpoint) if checkpoint and checkpoint.is_file() else None
        expected_hash = (config.get("initialization", {}).get(f"{name}_sha256")
                         if name in {"dsifn", "s2looking"} else None)
        if expected_hash:
            details["checkpoint_hash_expected"] = expected_hash
            details["checkpoint_hash_verified"] = checkpoint_hash == expected_hash
        teachers.append(
            TeacherPlugin(
                TeacherMetadata(
                    name=name,
                    role=role,
                    status=status,
                    dataset_root=str(root) if root else None,
                    checkpoint=str(checkpoint) if checkpoint else None,
                    checkpoint_sha256=checkpoint_hash,
                    outputs=available_outputs,
                    details=details,
                )
            )
        )
    return TeacherRegistry(teachers)
