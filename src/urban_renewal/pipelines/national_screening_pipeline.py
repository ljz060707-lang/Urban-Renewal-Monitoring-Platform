from __future__ import annotations


class NationalConstructionScreening:
    def __init__(self, random_state=20260729):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)

    def fit(self, features, labels, *, label_sources):
        accepted = {"official_ground_truth", "human_verified", "human_corrected"}
        if any(source not in accepted for source in label_sources):
            raise ValueError("national screening may train only on auditable formal labels")
        self.model.fit(features, labels); return self
