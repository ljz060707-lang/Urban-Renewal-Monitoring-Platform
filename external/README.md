# Optional external repositories

Clone external projects outside this starter so their histories and licenses stay separate.

```bash
mkdir -p ../external-repos
cd ../external-repos
git clone https://github.com/Robertgao0818/ZAsolar.git
git clone https://github.com/Robertgao0818/solar_backdating.git
git clone https://github.com/Meize0729/RSBuilding.git
git clone https://github.com/SebastianHafner/ContUrbanCD.git
git clone https://github.com/CosmiQ/CosmiQ_SN7_Baseline.git
git clone https://github.com/SpaceNetChallenge/SpaceNet7_Multi-Temporal_Solutions.git
```

Suggested order:

1. Run this starter with local building ground truth.
2. Compare the starter Mask R-CNN teacher with RSBuilding inference.
3. Keep the model with better local holdout performance or ensemble their probability masks.
4. Export reviewed aligned masks to a ContUrbanCD-compatible sequence dataset.
