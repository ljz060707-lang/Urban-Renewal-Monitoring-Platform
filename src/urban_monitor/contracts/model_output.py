"""Canonical prediction contract; unsupported outputs are explicit, never fake zeros."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputStatus(str, Enum):
    AVAILABLE = "available"
    PENDING = "pending"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


@dataclass
class OutputField:
    status: OutputStatus
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def available(cls, value: Any, **metadata: Any) -> "OutputField":
        if value is None:
            raise ValueError("available output requires a value")
        return cls(OutputStatus.AVAILABLE, value, metadata)

    @classmethod
    def unsupported(cls, reason: str) -> "OutputField":
        return cls(OutputStatus.UNSUPPORTED, None, {"reason": reason})

    @classmethod
    def pending(cls, reason: str) -> "OutputField":
        return cls(OutputStatus.PENDING, None, {"reason": reason})


@dataclass
class CanonicalModelOutput:
    any_change_probability: OutputField
    new_building_probability: OutputField
    demolition_probability: OutputField
    vacant_land_probability: OutputField
    construction_probability: OutputField
    building_probability: OutputField
    semantic_t1: OutputField
    semantic_t2: OutputField
    uncertainty: OutputField
    building_mask: OutputField
    building_boundary: OutputField

    @classmethod
    def unsupported(cls, supported: dict[str, Any], *, model_id: str) -> "CanonicalModelOutput":
        values = {}
        for name in cls.__dataclass_fields__:
            if name in supported:
                values[name] = OutputField.available(supported[name], model_id=model_id)
            else:
                values[name] = OutputField.unsupported(f"{model_id} does not provide {name}")
        return cls(**values)

    def available_heads(self) -> list[str]:
        return [name for name in self.__dataclass_fields__ if getattr(self, name).status == OutputStatus.AVAILABLE]
