"""Code-level holdout protection shared by onboarding and future training APIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


PROTECTED_PHASES = {"training", "validation", "threshold_tuning", "model_selection", "qualitative_review"}
FINAL_TEST_PHASE = "final_test"


@dataclass(frozen=True)
class TestAccessPolicy:
    dataset_id: str
    test_status: str
    locked_paths: tuple[str, ...] = ()

    @property
    def final_test_available(self) -> bool:
        return self.test_status == "locked_holdout"


class TestAccessGuard:
    __test__ = False

    def __init__(self, policies: list[TestAccessPolicy], *, phase: str, allow_final_test: bool = False, audit_path: str | Path | None = None):
        self.policies = {policy.dataset_id.casefold(): policy for policy in policies}
        self.phase = phase
        self.allow_final_test = bool(allow_final_test)
        self.audit_path = Path(audit_path) if audit_path else None
        self.read_count = 0
        self.events: list[dict] = []

    def check(self, dataset_id: str, split: str, path: str | Path | None = None) -> None:
        if str(split).casefold() != "test":
            return
        policy = self.policies.get(dataset_id.casefold())
        if policy is None:
            raise PermissionError(f"No Test policy registered for dataset={dataset_id}")
        if policy.test_status == "spent_test":
            self._record(dataset_id, path, "blocked_spent_test")
            raise PermissionError(f"{dataset_id} Test is SPENT and cannot be read again")
        if self.phase in PROTECTED_PHASES:
            self._record(dataset_id, path, f"blocked_phase_{self.phase}")
            raise PermissionError(f"{dataset_id} Test is forbidden during {self.phase}")
        if self.phase != FINAL_TEST_PHASE or not self.allow_final_test:
            self._record(dataset_id, path, "blocked_missing_explicit_final_test")
            raise PermissionError(f"{dataset_id} Test requires phase=final_test and --allow-final-test")
        if not policy.final_test_available:
            self._record(dataset_id, path, "blocked_policy")
            raise PermissionError(f"{dataset_id} Test policy does not permit final evaluation")
        self.read_count += 1
        self._record(dataset_id, path, "allowed_final_test")

    def _record(self, dataset_id: str, path: str | Path | None, decision: str) -> None:
        self.events.append({
            "dataset_id": dataset_id, "phase": self.phase, "path": str(path or ""),
            "decision": decision, "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if self.audit_path:
            self.write_audit()

    def write_audit(self) -> dict:
        payload = {
            "phase": self.phase, "allow_final_test": self.allow_final_test,
            "test_read_count": self.read_count, "policies": [asdict(value) for value in self.policies.values()],
            "events": self.events,
        }
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload


def default_policies() -> list[TestAccessPolicy]:
    return [
        TestAccessPolicy("dsifn", "spent_test", ("D:/Data Set/DSIFN-Dataset/test",)),
        TestAccessPolicy("s2looking", "spent_test", ("D:/Data Set/S2Looking/test",)),
        TestAccessPolicy("second", "locked_holdout", ("D:/Data Set/SECOND/test",)),
        TestAccessPolicy("whu_building", "locked_holdout", ("D:/Data Set/WHU_Building/test",)),
    ]
