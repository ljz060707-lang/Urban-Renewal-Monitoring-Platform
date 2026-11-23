# Model and data roles

B0: original FoundationEncoderDecoder/Swin-T, WHU historical Building supervision. Retained Xi'an roof candidate baseline, original >0.5 comparator, native256/stride192 probability blending. Target-domain roof semantic truth is not inferred from pretraining labels.

E3: original step120 S state, BANDON Roof and projected geometric Facade auxiliary plus native BONAI Roof. Two independent sigmoid outputs; no Background head, no Roof-minus-Facade. Diagnostic only. Recent 5000/10000-step S best0 experiments remain stopped/unpromoted.

V2 M_CORR step3634: original common/instance synthetic displacement protocol and native support. HR256/LR64, /255 RGB, original temperature/support/hypotheses. Complete-instance research movement only. Wrong-reference rejection and absolute position remain uncertified. The later 33-parameter negative-BCE experiment is not selected.

R4: existing mutual-best/overlap/shape/neighborhood rules, no trained identity network. R5 TBIA handcrafted prototype remains a diagnostic; no trained TBIA is invented. Candidate IDs never enter physical_id. Active B0 source-crop rechecks have fixed budgets and are not independent truth.

Areas: exact metric polygon union intersections, full-cell denominators, complete affine inherited from supplied target grid. Pure translation conserves area. Explicit frame-bound planar geometry is implemented; absent camera/roof-plane evidence leaves physical correction null. Surface and ground-footprint areas remain null.

Runtime: C:/Users/ljz52/.conda/envs/rsbuilding-win/python.exe (Python3.9). GPU modules are independent serial subprocesses; no optimizer, cloud, model download, or prediction GT argument. Exact source/runtime/checkpoint SHA bindings are in MODEL_ROUTING_MANIFEST.json. The C/D repository mirror is hardlinked; historical source was copied into an independent F-drive execution snapshot before adaptation.

Real demonstration: six distinct overlapping input crops from registered B01 development imagery, three actual acquisition dates recovered in the frozen tile-date ledger. Original log paths are missing, explicitly inherited evidence rather than newly re-read logs. It is a source-backed research demonstration, not independent change accuracy validation.
