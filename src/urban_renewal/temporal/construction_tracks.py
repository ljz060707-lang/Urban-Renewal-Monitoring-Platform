from __future__ import annotations

from datetime import date


def summarize_track(track_id, observations):
    ordered = sorted(observations, key=lambda item: item["observation_date"])
    dates = [date.fromisoformat(item["observation_date"]) for item in ordered]
    return {
        "track_id": track_id,
        "object_type": ordered[0].get("object_type", "construction_site"),
        "first_detected_date": ordered[0]["observation_date"],
        "last_detected_date": ordered[-1]["observation_date"],
        "initial_state": ordered[0]["construction_state"],
        "final_state": ordered[-1]["construction_state"],
        "state_sequence": [item["construction_state"] for item in ordered],
        "scene_type_sequence": [item.get("scene_type", "unknown_scene") for item in ordered],
        "duration_days": (dates[-1] - dates[0]).days,
        "max_site_area_m2": max((item.get("site_envelope_area_m2") or 0.0) for item in ordered),
        "mean_confidence": sum(item.get("confidence", 0.0) for item in ordered) / len(ordered),
        "minimum_confidence": min(item.get("confidence", 0.0) for item in ordered),
        "human_review_status": "pending",
        "scientific_status": "candidate_not_ground_truth",
    }
