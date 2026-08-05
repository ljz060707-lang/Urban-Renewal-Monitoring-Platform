# V2 Full Training Readiness

## Executed run

`20260726_184251` completed one real CUDA Quick epoch after the readiness patch. It used only train and validation records.

## Readiness patch evidence

- V2 source sampling remains DSIFN/S2Looking/SECOND = 1.0/2.0/1.5. S2Looking allocation is now direction-stratified with real demolition/new/ordinary images, then change-aware crops.
- Effective patches: DSIFN 21, S2Looking 42, SECOND 29; new-positive 32, demolition-positive 20, directional-positive fraction 0.5652. The prior Quick had only 2 demolition-positive patches.
- Derived entropy freezes the uncertainty head: `uncertainty_trainable=false`.
- Loss reporting decomposes BCE, Dice, boundary and semantic T1/T2 values. The semantic validation partition is a deterministic 10% partition of SECOND train, not a test split.
- `TEST_SAFETY_AUDIT.json` reports 0 test image reads for all three sources.

## Actual smoke metrics

- Total loss: 4.544857; gradient norm: 10.301141; NaN/Inf: none.
- Semantic validation: T1 pixel accuracy 0.00294 / mIoU 0.00107; T2 0.000145 / 0.000081; transition accuracy 0.03611. These low random-head smoke values are expected after one epoch and are not a quality claim.
- Validation PR-AUC: binary 0.23267, new 0.00865, demolition 0.000729.
- Peak CUDA allocation: 902,611,456 bytes.

## Decision

**Do not start Full training yet.** The engineering readiness gates now pass, but the one-epoch model quality signal—especially new/demolition PR-AUC—does not justify a 20-epoch run. First inspect the validation-only threshold report and run a short 2–3 epoch controlled V2 smoke with this sampler; test remains prohibited.
