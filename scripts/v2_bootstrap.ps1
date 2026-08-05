$ErrorActionPreference = "Stop"

# ============================================================
# Urban Renewal V2 bootstrap
#
# SAFE RULES:
# - Never read DSIFN old Test.
# - Never read S2Looking old Test.
# - SECOND Test is locked as future holdout.
# - SECOND mapping is NEVER guessed.
# - RSBuilding remains an external geometry teacher.
# ============================================================

$Repo = (Resolve-Path ".").Path

$DataRoot   = "D:\Data Set"
$DSIFN      = Join-Path $DataRoot "DSIFN-Dataset"
$S2Looking  = Join-Path $DataRoot "S2Looking"
$SECOND     = Join-Path $DataRoot "SECOND"
$RSBuilding = Join-Path $DataRoot "RSBuilding"

$Config = Join-Path $Repo "configs\change_public_v2_multisource_windows.yaml"

$AuditDir = Join-Path $Repo "outputs\v2_readiness"
New-Item -ItemType Directory -Force $AuditDir | Out-Null

Write-Host ""
Write-Host "============================================================"
Write-Host "URBAN RENEWAL V2 BOOTSTRAP"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# 1. Basic paths
# ============================================================

function Check-Path {
    param(
        [string]$Name,
        [string]$Path
    )

    if (Test-Path $Path) {
        Write-Host "[READY]   $Name"
        Write-Host "          $Path"
        return $true
    }
    else {
        Write-Host "[MISSING] $Name"
        Write-Host "          $Path"
        return $false
    }
}

$dsifnOK = Check-Path "DSIFN" $DSIFN
$s2OK    = Check-Path "S2Looking" $S2Looking
$secOK   = Check-Path "SECOND" $SECOND
$rsbOK   = Check-Path "RSBuilding" $RSBuilding

if (-not $dsifnOK) {
    throw "DSIFN dataset missing."
}

if (-not $s2OK) {
    throw "S2Looking dataset missing."
}

if (-not $secOK) {
    throw "SECOND dataset missing."
}

if (-not $rsbOK) {
    throw "RSBuilding directory missing."
}

# ============================================================
# 2. Verify frozen DSIFN checkpoint
# ============================================================

$DSIFNCheckpoint = Join-Path `
    $Repo `
    "outputs\pretraining\change\20260725_232317\checkpoints\best_binary_change.pth"

$ExpectedDSIFNHash = `
"d458e9b6bed969fcf79a38603608834b5bcafd8c055ba6b61ffcba8387819210"

if (-not (Test-Path $DSIFNCheckpoint)) {
    throw "Frozen DSIFN checkpoint missing: $DSIFNCheckpoint"
}

$actualHash = (
    Get-FileHash `
        $DSIFNCheckpoint `
        -Algorithm SHA256
).Hash.ToLower()

Write-Host ""
Write-Host "DSIFN checkpoint:"
Write-Host $DSIFNCheckpoint
Write-Host "SHA256:"
Write-Host $actualHash

if ($actualHash -ne $ExpectedDSIFNHash) {
    throw "DSIFN checkpoint SHA256 mismatch."
}

Write-Host "[PASS] Frozen DSIFN checkpoint verified."

# ============================================================
# 3. Locate frozen S2Looking V1 checkpoint
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "LOCATE S2LOOKING CHECKPOINT"
Write-Host "============================================================"

$S2SearchRoots = @(
    (Join-Path $Repo "outputs\models\urban_change_v1_s2looking"),
    (Join-Path $Repo "outputs\pretraining\change")
)

$S2Candidates = @()

foreach ($root in $S2SearchRoots) {

    if (-not (Test-Path $root)) {
        continue
    }

    $found = Get-ChildItem `
        $root `
        -Recurse `
        -File `
        -Include "*.pth","*.pt","*.ckpt" `
        -ErrorAction SilentlyContinue

    $S2Candidates += $found
}

if ($S2Candidates.Count -eq 0) {
    throw @"
No S2Looking checkpoint found.

Searched:
$($S2SearchRoots -join "`n")

Do NOT retrain yet.
Inspect outputs/models and outputs/pretraining/change first.
"@
}

# ------------------------------------------------------------
# Scoring:
# Prefer archived V1 + best + directional.
# Avoid DSIFN checkpoint.
# ------------------------------------------------------------

$Scored = foreach ($file in $S2Candidates) {

    $path = $file.FullName.ToLower()
    $name = $file.Name.ToLower()

    # Never accidentally select DSIFN
    if (
        $path -match "20260725_232317" -or
        $path -match "dsifn"
    ) {
        continue
    }

    $score = 0

    if ($path -match "urban_change_v1_s2looking") {
        $score += 100
    }

    if ($path -match "20260721_232528") {
        $score += 80
    }

    if ($name -match "best") {
        $score += 40
    }

    if (
        $name -match "direction" -or
        $name -match "new" -or
        $name -match "demolition"
    ) {
        $score += 30
    }

    # Prefer larger real model checkpoints
    if ($file.Length -gt 100MB) {
        $score += 10
    }

    [PSCustomObject]@{
        Score        = $score
        FullName     = $file.FullName
        Length       = $file.Length
        LastWriteTime = $file.LastWriteTime
    }
}

$Scored = $Scored |
    Sort-Object `
        Score, `
        LastWriteTime `
        -Descending

Write-Host ""
Write-Host "S2Looking checkpoint candidates:"
Write-Host ""

$Scored |
Select-Object -First 15 |
Format-Table `
    Score,
    Length,
    LastWriteTime,
    FullName `
    -AutoSize

if ($Scored.Count -eq 0) {
    throw "No usable S2Looking checkpoint found after filtering."
}

$S2Checkpoint = (
    $Scored |
    Select-Object -First 1
).FullName

Write-Host ""
Write-Host "[SELECTED] S2Looking checkpoint:"
Write-Host $S2Checkpoint

$S2Hash = (
    Get-FileHash `
        $S2Checkpoint `
        -Algorithm SHA256
).Hash.ToLower()

Write-Host "SHA256:"
Write-Host $S2Hash

# Persist selection for audit
$S2Selection = [PSCustomObject]@{
    checkpoint = $S2Checkpoint
    sha256     = $S2Hash
    selection_rule = "prefer archived V1 / baseline / best directional checkpoint"
}

$S2Selection |
ConvertTo-Json -Depth 5 |
Set-Content `
    (Join-Path $AuditDir "S2LOOKING_CHECKPOINT_SELECTION.json") `
    -Encoding UTF8

Write-Host "[PASS] S2Looking checkpoint selected."

# ============================================================
# 4. Audit SECOND train/validation mapping
#
# IMPORTANT:
# Only Train/Val labels are inspected.
# Test images are NEVER opened.
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "SECOND TRAIN/VAL MAPPING AUDIT"
Write-Host "============================================================"

$env:SECOND_ROOT = $SECOND
$env:V2_AUDIT_DIR = $AuditDir

@'
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


root = Path(os.environ["SECOND_ROOT"])
out = Path(os.environ["V2_AUDIT_DIR"])
out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Never inspect files under Test.
# ------------------------------------------------------------

TEST_NAMES = {
    "test",
    "testing",
}

TRAIN_VAL_NAMES = {
    "train",
    "training",
    "val",
    "valid",
    "validation",
}


def path_parts_lower(p: Path):
    return {x.lower() for x in p.parts}


def is_test_path(p: Path):
    return bool(
        path_parts_lower(p)
        &
        TEST_NAMES
    )


def is_train_val_path(p: Path):
    return bool(
        path_parts_lower(p)
        &
        TRAIN_VAL_NAMES
    )


# ------------------------------------------------------------
# Dataset structure
# ------------------------------------------------------------

top = []

for p in root.iterdir():

    if p.is_dir():

        count = sum(
            1
            for _ in p.rglob("*")
            if _.is_file()
        )

        top.append({
            "name": p.name,
            "file_count": count,
        })

(out / "SECOND_STRUCTURE.json").write_text(
    json.dumps(
        top,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Find possible documentation/mapping files
# ------------------------------------------------------------

doc_patterns = [
    "*.txt",
    "*.json",
    "*.yaml",
    "*.yml",
    "README*",
    "readme*",
    "*.md",
]

docs = []

for pat in doc_patterns:

    for p in root.rglob(pat):

        if is_test_path(p):
            continue

        if p.is_file():
            docs.append(str(p))

(out / "SECOND_MAPPING_DOCUMENT_CANDIDATES.json").write_text(
    json.dumps(
        sorted(set(docs)),
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------
# Find TRAIN/VAL label candidates only
# ------------------------------------------------------------

image_extensions = {
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
}

label_files = []

for p in root.rglob("*"):

    if not p.is_file():
        continue

    if p.suffix.lower() not in image_extensions:
        continue

    if is_test_path(p):
        continue

    if not is_train_val_path(p):
        continue

    lower = str(p).lower()

    # Candidate semantic labels.
    if any(
        token in lower
        for token in [
            "label",
            "mask",
            "gt",
            "semantic",
        ]
    ):
        label_files.append(p)


print(
    "Train/Val label candidates:",
    len(label_files)
)

# Use max 100 files for palette audit.
sample_files = label_files[:100]

rgb_counter = Counter()
id_counter = Counter()

read_count = 0

for p in sample_files:

    try:

        img = Image.open(p)
        arr = np.asarray(img)

        read_count += 1

        if arr.ndim == 2:

            values, counts = np.unique(
                arr,
                return_counts=True,
            )

            for value, count in zip(
                values,
                counts,
            ):

                id_counter[
                    int(value)
                ] += int(count)

        elif (
            arr.ndim == 3
            and
            arr.shape[2] >= 3
        ):

            rgb = arr[:, :, :3]

            flat = rgb.reshape(
                -1,
                3,
            )

            values, counts = np.unique(
                flat,
                axis=0,
                return_counts=True,
            )

            for value, count in zip(
                values,
                counts,
            ):

                key = tuple(
                    int(x)
                    for x in value
                )

                rgb_counter[
                    key
                ] += int(count)

    except Exception as e:

        print(
            "[WARN]",
            p,
            e,
        )


# ------------------------------------------------------------
# Export palette audit
# ------------------------------------------------------------

rgb_csv = (
    out
    /
    "SECOND_TRAINVAL_RGB_PALETTE.csv"
)

with rgb_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "r",
        "g",
        "b",
        "pixel_count",
    ])

    for (
        r,
        g,
        b,
    ), count in rgb_counter.most_common():

        writer.writerow([
            r,
            g,
            b,
            count,
        ])


id_csv = (
    out
    /
    "SECOND_TRAINVAL_CLASS_IDS.csv"
)

with id_csv.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "pixel_count",
    ])

    for value, count in id_counter.most_common():

        writer.writerow([
            value,
            count,
        ])


audit = {
    "second_root": str(root.resolve()),

    "test_image_reads": 0,

    "train_validation_label_files_found":
        len(label_files),

    "train_validation_label_files_sampled":
        read_count,

    "unique_rgb_colors":
        len(rgb_counter),

    "unique_class_ids":
        len(id_counter),

    "mapping_verified":
        False,

    "building_class_id":
        None,

    "policy": (
        "Do not derive SECOND building/new/"
        "demolition labels until mapping is "
        "manually verified."
    ),
}

(out / "SECOND_MAPPING_AUDIT.json").write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("SECOND mapping audit complete.")
print("Test image reads: 0")
print("RGB palette:", rgb_csv)
print("Class IDs :", id_csv)
'@ | python -

# ============================================================
# 5. Lock SECOND Test as V2 holdout
#
# ONLY filenames are enumerated.
# Raster/image contents are not opened.
# ============================================================

Write-Host ""
Write-Host "Creating V2 holdout protocol..."

$env:SECOND_ROOT = $SECOND
$env:V2_AUDIT_DIR = $AuditDir

@'
from pathlib import Path
import hashlib
import json
import os


root = Path(
    os.environ["SECOND_ROOT"]
)

out = Path(
    os.environ["V2_AUDIT_DIR"]
)


test_dirs = [
    p
    for p in root.iterdir()
    if p.is_dir()
    and p.name.lower() in {
        "test",
        "testing",
    }
]

ids = []

for test_dir in test_dirs:

    # Metadata only:
    # enumerate relative filenames,
    # NEVER open file contents.
    for p in test_dir.rglob("*"):

        if p.is_file():

            ids.append(
                str(
                    p.relative_to(root)
                ).replace(
                    "\\",
                    "/",
                )
            )


ids = sorted(ids)

payload = "\n".join(
    ids
).encode(
    "utf-8"
)

digest = hashlib.sha256(
    payload
).hexdigest()

protocol = {
    "dataset": "SECOND",

    "split": "test",

    "file_count_metadata_only":
        len(ids),

    "sample_ids_sha256":
        digest,

    "created_before_v2_training":
        True,

    "image_contents_read":
        False,

    "used_for_training":
        False,

    "used_for_validation":
        False,

    "used_for_threshold_selection":
        False,

    "policy":
        "Locked future V2 holdout. "
        "Do not inspect images during development.",
}

path = (
    out
    /
    "V2_HOLDOUT_PROTOCOL.json"
)

path.write_text(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        protocol,
        ensure_ascii=False,
        indent=2,
    )
)
'@ | python -

# ============================================================
# 6. Audit RSBuilding
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "RSBUILDING AUDIT"
Write-Host "============================================================"

$RSRepoCandidates = @(
    (Join-Path $RSBuilding "repo"),
    $RSBuilding
)

$RSRepo = $null

foreach ($candidate in $RSRepoCandidates) {

    if (
        (Test-Path (Join-Path $candidate "configs")) -or
        (Test-Path (Join-Path $candidate "opencd"))
    ) {

        $RSRepo = $candidate
        break
    }
}

if ($null -eq $RSRepo) {

    Write-Host "[WARN] RSBuilding repo not recognized."

}
else {

    Write-Host "[READY] RSBuilding repo:"
    Write-Host $RSRepo

}

$RSWeights = Get-ChildItem `
    $RSBuilding `
    -Recurse `
    -File `
    -Include "*.pth","*.pt","*.ckpt" |
Sort-Object Length -Descending

Write-Host ""
Write-Host "RSBuilding checkpoint candidates:"

if ($RSWeights.Count -eq 0) {

    Write-Host "[MISSING] no .pth/.pt/.ckpt found"

}
else {

    $RSWeights |
    Select-Object -First 20 |
    Format-Table `
        FullName,
        Length,
        LastWriteTime `
        -AutoSize
}

# ============================================================
# 7. Check V2 CLI exists
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "CHECK V2 CLI"
Write-Host "============================================================"

$help = (
    python .\run.py --help 2>&1 |
    Out-String
)

$requiredCommands = @(
    "urban-renewal-preflight",
    "urban-renewal-infer",
    "change-all",
    "change-preflight"
)

foreach ($cmd in $requiredCommands) {

    if (
        $help -match [regex]::Escape(
            $cmd
        )
    ) {

        Write-Host "[READY] $cmd"

    }
    else {

        Write-Host "[MISSING] $cmd"

    }
}

if (
    $help -notmatch "urban-renewal-preflight"
) {

    throw @"
The previous V2 Codex refactor has not been fully integrated.
Do not start Full training.
First finish the plug-in V2 architecture.
"@
}

# ============================================================
# 8. Check V2 config
# ============================================================

if (-not (Test-Path $Config)) {

    throw "Missing V2 config: $Config"
}

Write-Host ""
Write-Host "[READY] V2 config:"
Write-Host $Config

# ============================================================
# 9. Patch LOCAL ROOTS ONLY
#
# Does NOT guess SECOND mapping.
# ============================================================

$env:V2_CONFIG = $Config
$env:DSIFN_ROOT = $DSIFN
$env:S2_ROOT = $S2Looking
$env:SECOND_ROOT = $SECOND
$env:RSB_ROOT = $RSBuilding
$env:DSIFN_CKPT = $DSIFNCheckpoint
$env:S2_CKPT = $S2Checkpoint

@'
from pathlib import Path
import os

import yaml


config_path = Path(
    os.environ["V2_CONFIG"]
)

cfg = yaml.safe_load(
    config_path.read_text(
        encoding="utf-8"
    )
)

if cfg is None:
    cfg = {}


datasets = cfg.setdefault(
    "datasets",
    {}
)

datasets.setdefault(
    "dsifn",
    {}
)

datasets.setdefault(
    "s2looking",
    {}
)

datasets.setdefault(
    "second",
    {}
)


datasets["dsifn"].update({
    "enabled": True,
    "root":
        os.environ["DSIFN_ROOT"]
        .replace("\\", "/"),
})

datasets["s2looking"].update({
    "enabled": True,
    "root":
        os.environ["S2_ROOT"]
        .replace("\\", "/"),
})

second = datasets["second"]

second.update({
    "enabled": True,
    "root":
        os.environ["SECOND_ROOT"]
        .replace("\\", "/"),
})

# Never guess mapping.
second.setdefault(
    "building_class_id",
    None,
)

second.setdefault(
    "class_mapping_file",
    None,
)

second.setdefault(
    "mapping_verified",
    False,
)


teachers = cfg.setdefault(
    "teachers",
    {}
)

teachers.setdefault(
    "rsbuilding",
    {}
)

teachers["rsbuilding"].update({
    "enabled": True,
    "root":
        os.environ["RSB_ROOT"]
        .replace("\\", "/"),
    "external_inference": True,
})


init = cfg.setdefault(
    "initialization",
    {}
)

init["dsifn_checkpoint"] = (
    os.environ["DSIFN_CKPT"]
    .replace("\\", "/")
)

init["s2looking_checkpoint"] = (
    os.environ["S2_CKPT"]
    .replace("\\", "/")
)


training = cfg.setdefault(
    "training",
    {}
)

training.setdefault(
    "epochs",
    20,
)

training.setdefault(
    "dataset_sampling",
    {
        "s2looking": 2.0,
        "second": 1.5,
        "dsifn": 1.0,
    },
)


config_path.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)

print(
    "Patched:",
    config_path,
)

print(
    "SECOND mapping_verified:",
    second.get(
        "mapping_verified"
    ),
)

print(
    "SECOND building_class_id:",
    second.get(
        "building_class_id"
    ),
)
'@ | python -

# ============================================================
# 10. Run urban-renewal-preflight
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "URBAN RENEWAL PREFLIGHT"
Write-Host "============================================================"

python .\run.py urban-renewal-preflight

if ($LASTEXITCODE -ne 0) {
    throw "urban-renewal-preflight failed."
}

# ============================================================
# 11. Run V2 dataset preflight
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "V2 CHANGE PREFLIGHT"
Write-Host "============================================================"

python .\run.py change-preflight `
    --config $Config

if ($LASTEXITCODE -ne 0) {
    throw "V2 change-preflight failed."
}

Write-Host ""
Write-Host "============================================================"
Write-Host "BOOTSTRAP COMPLETE"
Write-Host "============================================================"

Write-Host ""
Write-Host "IMPORTANT:"
Write-Host "1. DSIFN old Test was NOT read."
Write-Host "2. S2Looking Test was NOT read."
Write-Host "3. SECOND Test contents were NOT read."
Write-Host "4. SECOND mapping was NOT guessed."
Write-Host "5. RSBuilding remains external."
Write-Host ""
Write-Host "NEXT: inspect SECOND mapping audit, then run V2 Quick."
