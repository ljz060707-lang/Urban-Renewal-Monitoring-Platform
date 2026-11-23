from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from create_xian_publication_figures import (
    BBOX,
    COLORS,
    LATEST,
    RAW,
    bbox_tuple,
    blend_mask,
    read_label,
    read_rgb,
)


OUT = Path(r"F:\xian\西安_当前最佳候选交付_20260917\09_clean_transition_maps_20260917")


def render_clean_map(interval: str, base_year: str, filename: str) -> Path:
    image, _ = read_rgb(RAW[base_year], bbox_tuple(), target_width=2600)
    stable, _ = read_label(LATEST["states"][interval]["STABLE"], bbox_tuple(), target_width=2600)
    unresolved, _ = read_label(LATEST["states"][interval]["UNKNOWN_CONFLICT"], bbox_tuple(), target_width=2600)
    addition, _ = read_label(LATEST["states"][interval]["ADDITION"], bbox_tuple(), target_width=2600)
    demolition, _ = read_label(LATEST["states"][interval]["DEMOLITION"], bbox_tuple(), target_width=2600)

    # Keep the same three-class display convention as the reviewed figure:
    # unresolved/conflict is folded into Stable for display only.
    composite = blend_mask(image, (stable > 0) | (unresolved > 0), COLORS["stable"], alpha=0.33)
    composite = blend_mask(composite, addition > 0, COLORS["addition"], alpha=0.52)
    composite = blend_mask(composite, demolition > 0, COLORS["demolition"], alpha=0.55)

    height, width = composite.shape[:2]
    fig = plt.figure(figsize=(8.0, 8.0 * height / width), dpi=600, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(composite, origin="upper", interpolation="bilinear", aspect="auto")
    ax.set_axis_off()
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / filename
    fig.savefig(output, dpi=600, facecolor="black", edgecolor="none", pad_inches=0)
    plt.close(fig)
    return output


def main() -> None:
    outputs = [
        render_clean_map("2018_2019", "2019", "2018_2019_clean_transition_map.png"),
        render_clean_map("2019_2020", "2020", "2019_2020_clean_transition_map.png"),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
