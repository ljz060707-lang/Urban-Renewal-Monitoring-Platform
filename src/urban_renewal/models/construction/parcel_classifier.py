from __future__ import annotations


class ConstructionParcelClassifier:
    """Explainable sklearn parcel classifier with feature-importance export."""

    def __init__(self, random_state: int = 20260729):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(random_state=random_state)
        self.feature_names_: list[str] = []

    def fit(self, rows, labels):
        self.feature_names_ = list(rows[0])
        self.model.fit([[row[name] for name in self.feature_names_] for row in rows], labels)
        return self

    def predict_proba(self, rows):
        return self.model.predict_proba([[row[name] for name in self.feature_names_] for row in rows])
