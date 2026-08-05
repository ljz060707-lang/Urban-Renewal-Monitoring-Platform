from __future__ import annotations

import os
import sys
from pathlib import Path


RSBUILDING_ENV = Path(
    sys.executable
).resolve().parent


def _runtime_dll_directories() -> list[Path]:
    directories = [
        RSBUILDING_ENV / "Library" / "bin",
        (
            RSBUILDING_ENV
            / "Lib"
            / "site-packages"
            / "torch"
            / "lib"
        ),
    ]

    directories.extend(
        path.parent
        for path in RSBUILDING_ENV.rglob(
            "nvrtc-builtins64_117.dll"
        )
    )

    unique = []

    for directory in directories:
        directory = directory.resolve()

        if (
            directory.exists()
            and directory not in unique
        ):
            unique.append(directory)

    return unique


DLL_DIRECTORIES = _runtime_dll_directories()
DLL_HANDLES = []


for directory in DLL_DIRECTORIES:
    os.environ["PATH"] = (
        str(directory)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    if hasattr(os, "add_dll_directory"):
        DLL_HANDLES.append(
            os.add_dll_directory(
                str(directory)
            )
        )


if not any(
    directory.joinpath(
        "nvrtc-builtins64_117.dll"
    ).exists()
    for directory in DLL_DIRECTORIES
):
    raise RuntimeError(
        "RSBuilding NVRTC runtime is unavailable: "
        "nvrtc-builtins64_117.dll was not found."
    )