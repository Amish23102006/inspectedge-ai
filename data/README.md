# Data

No dataset files are committed. `data/raw/` is git-ignored.

## MVTec AD (vision MVP)

- Source: official MVTec website (search "MVTec AD download"); you must accept the licence there.
- Licence: **CC BY-NC-SA 4.0**, non-commercial use only. Cite the MVTec AD papers (Bergmann et al., CVPR 2019 and IJCV 2021).
- Content: more than 5,000 high-resolution images across 15 categories; defect-free training images; normal and anomalous test images; pixel-precise masks.

Download a single category archive (e.g. `bottle`) and extract so the layout is:

```
data/raw/mvtec_ad/
└── bottle/
    ├── train/good/*.png
    ├── test/good/*.png
    ├── test/<defect_type>/*.png
    └── ground_truth/<defect_type>/<name>_mask.png
```

Then run the `1. Prepare data` launch config (or `python -m inspectedge.data.prepare`). It writes `artefacts/manifests/<category>_manifest.csv` and a summary JSON with counts, missing masks and any cross-split duplicates. Record the archive's SHA-256 here once downloaded:

| Category | Archive | SHA-256 | Downloaded on |
|---|---|---|---|
| bottle | – | – | – |

### Why `bottle` by default?

It is a reasonable first category: an object (not a texture) with clear structural defects. Change `data.category` in `configs/default.yaml` and document your reasoning in the main README.

## Planned datasets (not used yet)

| Dataset | Use | Licence (verify at source) |
|---|---|---|
| MVTec AD 2 | Lighting-shift robustness, private evaluation server | CC BY-NC-SA 4.0 |
| VisA (AWS Open Data) | External validation | CC BY 4.0 |
| NASA CMAPSS | Remaining-useful-life prediction | NASA open data |

Never use confidential company images, process settings, specifications or defect records.
