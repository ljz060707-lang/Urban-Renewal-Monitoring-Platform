from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock


HEADS = ("any_change", "new_building", "demolition", "semantic_t1", "semantic_t2")


def supervision_matrix(*, second_mapping_verified: bool = False) -> dict[str, dict[str, bool]]:
    """Return the explicit dataset × head routing contract.

    A false value means "no target exists", never "use an all-zero target".
    """
    return {
        "dsifn": {
            "any_change": True,
            "new_building": False,
            "demolition": False,
            "semantic_t1": False,
            "semantic_t2": False,
        },
        "s2looking": {
            "any_change": True,
            "new_building": True,
            "demolition": True,
            "semantic_t1": False,
            "semantic_t2": False,
        },
        "second": {
            "any_change": True,
            "new_building": bool(second_mapping_verified),
            "demolition": bool(second_mapping_verified),
            "semantic_t1": True,
            "semantic_t2": True,
        },
    }


def active_heads(dataset_name: str, *, second_mapping_verified: bool = False) -> tuple[str, ...]:
    matrix = supervision_matrix(second_mapping_verified=second_mapping_verified)
    key = dataset_name.casefold()
    if key not in matrix:
        raise KeyError(f"Unknown V2 dataset: {dataset_name}")
    return tuple(head for head in HEADS if matrix[key][head])


@dataclass
class TestReadReceipt:
    command: str
    test_images_read: int
    allowed: bool


class TestReadGuard:
    """Single process-local gate for held-out image reads.

    Development commands construct the guard with ``allow_test=False`` and must
    call ``check`` before opening an image. Final-test is the only intended
    caller that can opt in.
    """

    _lock = Lock()
    _counts: dict[str, int] = {}
    __test__ = False

    def __init__(self, command: str, *, allow_test: bool = False):
        self.command = command
        self.allow_test = allow_test
        with self._lock:
            self._counts.setdefault(command, 0)

    def check(self, split: str, path: str | Path | None = None) -> None:
        if str(split).casefold() != "test":
            return
        if not self.allow_test:
            suffix = f": {path}" if path else ""
            raise PermissionError(f"{self.command} is not allowed to read split=test{suffix}")
        with self._lock:
            self._counts[self.command] += 1

    def receipt(self) -> TestReadReceipt:
        with self._lock:
            count = self._counts.get(self.command, 0)
        return TestReadReceipt(self.command, count, self.allow_test)

    @classmethod
    def reset(cls, command: str | None = None) -> None:
        with cls._lock:
            if command is None:
                cls._counts.clear()
            else:
                cls._counts.pop(command, None)
