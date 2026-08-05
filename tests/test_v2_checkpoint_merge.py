import json

import torch

import urban_monitor.change.checkpoint_merge_v2 as merge


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(2, 2)
        self.new_building_head = torch.nn.Linear(2, 1)
        self.demolition_head = torch.nn.Linear(2, 1)
        self.semantic_t1_head = torch.nn.Linear(2, 3)
        self.semantic_t2_head = torch.nn.Linear(2, 3)


def test_checkpoint_merge_has_explicit_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(merge, "build_change_model", lambda config: Tiny())
    ds, s2 = Tiny(), Tiny()
    for value in ds.parameters():
        value.data.fill_(1)
    for value in s2.parameters():
        value.data.fill_(2)
    ds_path, s2_path = tmp_path / "ds.pth", tmp_path / "s2.pth"
    torch.save({"model": ds.state_dict()}, ds_path)
    torch.save({"model": s2.state_dict()}, s2_path)
    output, report = tmp_path / "merged.pth", tmp_path / "report.json"
    result = merge.merge_v2_checkpoints({}, ds_path, s2_path, output, report)
    state = torch.load(output, weights_only=False)["model"]
    assert torch.all(state["shared.weight"] == 1)
    assert torch.all(state["new_building_head.weight"] == 2)
    assert result["status"] == "success"
    assert json.loads(report.read_text())["output_sha256"]

