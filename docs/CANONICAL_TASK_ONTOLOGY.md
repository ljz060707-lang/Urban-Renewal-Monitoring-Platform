# Canonical Task Ontology

## Layer 1 — Pixel / surface state

`background_or_other`, `vacant_land`, `construction_site`, `building`, `unknown`

## Layer 2 — Change direction

`any_change`, `new_building`, `demolition`

## Layer 3 — Semantic transition

由 semantic T1/T2 派生，例如 vacant→construction、construction→building、building→vacant。现在不训练虚构的 transition head。

## Layer 4 — Temporal urban-renewal state

`stable_vacant`, `construction_candidate`, `construction_active`, `newly_completed_building`, `stable_building`, `demolition_candidate`, `demolished`, `redevelopment_candidate`, `uncertain`

TFR/MTI 的 native labels 必须由 dataset-specific YAML 映射；未读到权威标签说明前不猜类别含义。
