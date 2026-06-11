# Data Sources

This directory should contain the following files before running any notebooks.
---

## Metadata CSVs (`data/raw/`)

| Filename | Description | Where to get it |
|----------|-------------|-----------------|
| `Cajal_BothGouwens_metadata.csv` | Cell-type labels for mouse interneurons (Gouwens 2019 + 2020) | Assembled from the Allen Cell Types Database — see below |
| `Cajal_mousehuman_metadatav2.csv` | Cell-type labels for mouse + human interneurons (Lee et al. 2023) | Assembled from the Allen Cell Types Database — see below |

### Required columns
Both files must contain:
- `Specimen ID` — matches SWC filenames (without `.swc` suffix)
- `T type sub-class` — interneuron sub-class label (e.g. Lamp5, Pvalb, Sst, Vip…)
- `T type sentencecase` — T-type label in sentence case
- `Dataset` — one of `Old Gouwen`, `New Patchseq`, `Human Patchseq`
- `Binary_Lamp5` — binary label: `Lamp5` or `Non-Lamp5`

---

## SWC Morphology Files (`data/swc/`)

### Mouse interneurons (`data/swc/Both_mouse_swc/`)
- **Source:** Allen Cell Types Database
- **Papers:** Gouwens et al. 2019 (*Nature Neuroscience*) and Gouwens et al. 2020 (*Cell*)
- **Download:** https://celltypes.brain-map.org/data
  - Filter by: Species = Mouse, Data type = Morphology
  - Download reconstructions in SWC format

### Mouse + Human interneurons (`data/swc/only_metadata_swc/`)
- **Mouse cells:** same Allen Cell Types Database as above
- **Human cells:** Lee et al. 2023 (*Nature*)
  - https://celltypes.brain-map.org/data (filter: Species = Human)
  - Or directly from: https://github.com/AllenInstitute/patchseq_human_L23

---

## Preprocessed outputs (`data/processed/`)

These are generated automatically by the notebook and do not need to be downloaded:
- `noaxons_both_mouse_100pts_euclidean_icdm.csv`
- `noaxons_both_mouse_100pts_euclidean_GW_dmat.csv`
- `mousehuman_100pts_euclidean_icdm.csv`
- `mousehuman_100pts_euclidean_GW_dmat.csv`

> The GW distance matrix computation is slow (~2–8 hours depending on hardware).
> Pre-computed matrices can be shared via Zenodo or Google Drive — contact the author.

---

## Environment setup

```bash
conda create -n morphoL python=3.9
conda activate morphoL
pip install cajal umap-learn scikit-learn seaborn plotly pandas numpy scipy
```
