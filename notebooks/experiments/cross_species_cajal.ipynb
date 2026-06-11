# %% [markdown]
# # Cross-Species Neuron Morphology — CAJAL Pipeline
# **Author:** Nandita Jayadevan  
# **Project:** HMB496 Undergraduate Thesis, University of Toronto (2024)  
# **Supervisor:** Dr. Shreejoy Tripathy
#
# ## Reproducibility notes
#
# ### Data sources
# | File | Origin | Where to get it |
# |------|--------|-----------------|
# | Mouse SWC reconstructions (`Both_mouse_swc/`) | Allen Institute — Gouwens et al. 2019 & 2020 | https://celltypes.brain-map.org/data |
# | Mouse+Human SWC reconstructions (`only_metadata_swc/`) | Allen Institute — Lee et al. 2023 (human) + Gouwens mouse | https://celltypes.brain-map.org/data |
# | `Cajal_BothGouwens_metadata.csv` | Manually assembled from Allen Cell Types DB | See `data/raw/README.md` |
# | `Cajal_mousehuman_metadatav2.csv` | Manually assembled from Allen Cell Types DB | See `data/raw/README.md` |
#
# ### Environment
# ```
# conda create -n morphoL python=3.9
# conda activate morphoL
# pip install cajal umap-learn scikit-learn seaborn plotly pandas numpy scipy
# ```
#
# ### Directory layout expected
# ```
# cross_species_neuron_morphology_ml/
# ├── data/
# │   ├── raw/
# │   │   ├── Cajal_BothGouwens_metadata.csv
# │   │   └── Cajal_mousehuman_metadatav2.csv
# │   └── processed/          # auto-created below
# ├── notebooks/experiments/
# │   └── cross_species_cajal.py   ← this file
# └── results/
# ```

# %% [markdown]
# ## 0 — Imports & path setup

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import umap
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle as sk_shuffle

import cajal.sample_swc
import cajal.swc
import cajal.run_gw
import cajal.utilities
import cajal.laplacian_score

pio.renderers.default = "iframe"

# ── Resolve paths relative to this file so the notebook works from any clone ──
HERE = os.path.dirname(os.path.abspath(__file__))        # notebooks/experiments/
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))   # repo root

RAW_DIR       = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR   = os.path.join(ROOT, "results")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,   exist_ok=True)

# ── SWC folders — only these two lines need editing if data lives elsewhere ──
MOUSE_SWC_DIR      = os.path.join(ROOT, "data", "swc", "Both_mouse_swc")
MOUSEHUMAN_SWC_DIR = os.path.join(ROOT, "data", "swc", "only_metadata_swc")

# Verify raw data files exist before running
for f in [
    os.path.join(RAW_DIR, "Cajal_BothGouwens_metadata.csv"),
    os.path.join(RAW_DIR, "Cajal_mousehuman_metadatav2.csv"),
]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"\nMissing data file: {f}\n"
            "See the 'Data sources' table at the top of this notebook for download links."
        )

print("✓ All data files found.")
print(f"  ROOT        : {ROOT}")
print(f"  PROCESSED   : {PROCESSED_DIR}")
print(f"  Mouse SWCs  : {MOUSE_SWC_DIR}")
print(f"  M+H SWCs    : {MOUSEHUMAN_SWC_DIR}")


# %% [markdown]
# ---
# ## PART 1 — Both-Mouse Datasets (Gouwens 2019 + 2020)

# %% [markdown]
# ### 1.1 — Compute ICDM (intra-cell distance matrix)

# %%
MOUSE_ICDM = os.path.join(PROCESSED_DIR, "noaxons_both_mouse_100pts_euclidean_icdm.csv")

if os.path.exists(MOUSE_ICDM):
    print(f"ICDM already exists, skipping recomputation:\n  {MOUSE_ICDM}")
else:
    cajal.sample_swc.compute_icdm_all_euclidean(
        infolder=MOUSE_SWC_DIR,
        out_csv=MOUSE_ICDM,
        preprocess=cajal.swc.preprocessor_eu(
            structure_ids=[1, 3, 4],   # soma=1, basal dendrite=3, apical dendrite=4
            soma_component_only=False,
        ),
        n_sample=100,
        num_processes=8,
    )
    print(f"✓ ICDM saved to {MOUSE_ICDM}")


# %% [markdown]
# ### 1.2 — Compute Gromov–Wasserstein distance matrix

# %%
MOUSE_GW = os.path.join(PROCESSED_DIR, "noaxons_both_mouse_100pts_euclidean_GW_dmat.csv")

if os.path.exists(MOUSE_GW):
    print(f"GW matrix already exists, skipping:\n  {MOUSE_GW}")
else:
    cajal.run_gw.compute_gw_distance_matrix(
        MOUSE_ICDM,
        MOUSE_GW,
        num_processes=8,
    )
    print(f"✓ GW distance matrix saved to {MOUSE_GW}")


# %% [markdown]
# ### 1.3 — Load GW matrix + metadata

# %%
cells_raw, gw_dist_dict = cajal.utilities.read_gw_dists(MOUSE_GW, header=True)
gw_dist_mouse = cajal.utilities.dist_mat_of_dict(gw_dist_dict)

# Strip alignment suffixes added during preprocessing
cells = [c.replace("_transformed", "").replace("_upright", "").replace("_m", "")
         for c in cells_raw]

metadata_mouse = pd.read_csv(os.path.join(RAW_DIR, "Cajal_BothGouwens_metadata.csv"))
metadata_mouse.index = [str(m) for m in metadata_mouse["Specimen ID"]]
metadata_mouse = metadata_mouse.loc[cells]

print(f"✓ Loaded {len(cells)} cells")
print(metadata_mouse["T type sub-class"].value_counts())


# %% [markdown]
# ### 1.4 — UMAP embedding

# %%
reducer = umap.UMAP(metric="precomputed", random_state=1)
embedding_mouse = reducer.fit_transform(gw_dist_mouse)

px.scatter(
    x=embedding_mouse[:, 0],
    y=embedding_mouse[:, 1],
    template="simple_white",
    hover_name=[c + ".swc" for c in cells],
    color=metadata_mouse["T type sub-class"],
    title="Mouse datasets — T type sub-class",
    width=800, height=600,
)


# %% [markdown]
# ### 1.5 — Leiden clustering

# %%
clusters_mouse = cajal.utilities.leiden_clustering(gw_dist_mouse, seed=1)

px.scatter(
    x=embedding_mouse[:, 0],
    y=embedding_mouse[:, 1],
    template="simple_white",
    hover_name=[c + ".swc" for c in cells],
    color=[str(m) for m in clusters_mouse],
    title="Mouse datasets — Leiden clusters",
    width=800, height=600,
)


# %% [markdown]
# ### 1.6 — Laplacian score (sub-class)

# %%
layers = np.unique(metadata_mouse["T type sub-class"])
indicator = (np.array(metadata_mouse["T type sub-class"])[:, None] == layers) * 1

laplacian_mouse = pd.DataFrame(
    cajal.laplacian_score.laplacian_scores(
        indicator,
        gw_dist_mouse,
        np.median(squareform(gw_dist_mouse)),
        permutations=5000,
        covariates=None,
        return_random_laplacians=False,
    )[0]
)
laplacian_mouse.index = layers
print(laplacian_mouse)


# %% [markdown]
# ### 1.7 — Random Forest: sub-class, 5-fold CV
#
# **Fix applied:** `gw_dist[np.ix_(mask, mask)]` filters both axes of the
# symmetric distance matrix. The original code only filtered rows, producing a
# non-square matrix that silently corrupted classifier inputs.

# %%
cell_type = np.array(metadata_mouse["T type sub-class"])
class_counts = pd.Series(cell_type).value_counts()
selected_classes = class_counts[class_counts >= 5].index

mask = metadata_mouse["T type sub-class"].isin(selected_classes).values   # bool array
filtered_gw   = gw_dist_mouse[np.ix_(mask, mask)]                         # FIX: both axes
filtered_labels = cell_type[mask]

clf = RandomForestClassifier(n_estimators=100, random_state=0)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

accuracy = cross_val_score(clf, X=filtered_gw, y=filtered_labels, cv=cv)
print(f"Mean CV accuracy (sub-class): {np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}")

preds = cross_val_predict(clf, X=filtered_gw, y=filtered_labels, cv=cv)

conf_mat = confusion_matrix(filtered_labels, preds, labels=selected_classes)
conf_pct  = conf_mat / conf_mat.sum(axis=1, keepdims=True)
conf_df   = pd.DataFrame(conf_pct, index=selected_classes, columns=selected_classes)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_df, annot=True, cmap="Reds", fmt=".2f", ax=ax)
ax.set_title("Mouse (Both Gouwens) — Sub-class RF Classifier")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mouse_subclass_confusion.png"), dpi=150)
plt.show()


# %% [markdown]
# ### 1.8 — Random Forest: T-type (sentence case), 5-fold CV

# %%
cell_type_t = np.array(metadata_mouse["T type sentencecase"])
counts_t = pd.Series(cell_type_t).value_counts()
selected_t = counts_t[counts_t >= 5].index

mask_t = metadata_mouse["T type sentencecase"].isin(selected_t).values
filtered_gw_t     = gw_dist_mouse[np.ix_(mask_t, mask_t)]
filtered_labels_t = cell_type_t[mask_t]

clf_t = RandomForestClassifier(n_estimators=100, random_state=0)
cv_t  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

accuracy_t = cross_val_score(clf_t, X=filtered_gw_t, y=filtered_labels_t, cv=cv_t)
print(f"Mean CV accuracy (T type): {np.mean(accuracy_t):.4f} ± {np.std(accuracy_t):.4f}")

preds_t = cross_val_predict(clf_t, X=filtered_gw_t, y=filtered_labels_t, cv=cv_t)

conf_mat_t = confusion_matrix(filtered_labels_t, preds_t, labels=selected_t)
conf_pct_t  = conf_mat_t / conf_mat_t.sum(axis=1, keepdims=True)
conf_df_t   = pd.DataFrame(conf_pct_t, index=selected_t, columns=selected_t)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_df_t, annot=True, cmap="Greens", fmt=".2f", ax=ax)
ax.set_title("Mouse (Both Gouwens) — T-type RF Classifier")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mouse_ttype_confusion.png"), dpi=150)
plt.show()


# %% [markdown]
# ---
# ## PART 2 — Mouse + Human Cross-Species

# %% [markdown]
# ### 2.1 — Compute ICDM

# %%
MH_ICDM = os.path.join(PROCESSED_DIR, "mousehuman_100pts_euclidean_icdm.csv")

if os.path.exists(MH_ICDM):
    print(f"ICDM already exists, skipping:\n  {MH_ICDM}")
else:
    cajal.sample_swc.compute_icdm_all_euclidean(
        infolder=MOUSEHUMAN_SWC_DIR,
        out_csv=MH_ICDM,
        preprocess=cajal.swc.preprocessor_eu(
            structure_ids=[1, 3, 4],
            soma_component_only=False,
        ),
        n_sample=100,
        num_processes=8,
    )
    print(f"✓ ICDM saved to {MH_ICDM}")


# %% [markdown]
# ### 2.2 — Compute GW distance matrix

# %%
MH_GW = os.path.join(PROCESSED_DIR, "mousehuman_100pts_euclidean_GW_dmat.csv")

if os.path.exists(MH_GW):
    print(f"GW matrix already exists, skipping:\n  {MH_GW}")
else:
    cajal.run_gw.compute_gw_distance_matrix(MH_ICDM, MH_GW, num_processes=8)
    print(f"✓ GW matrix saved to {MH_GW}")


# %% [markdown]
# ### 2.3 — Load GW matrix + metadata

# %%
mh_cells_raw, mh_gw_dict = cajal.utilities.read_gw_dists(MH_GW, header=True)
gw_dist_mh = cajal.utilities.dist_mat_of_dict(mh_gw_dict)

mh_cells = [c.replace("_transformed", "").replace("_upright", "").replace("_m", "")
            for c in mh_cells_raw]

metadata_mh = pd.read_csv(os.path.join(RAW_DIR, "Cajal_mousehuman_metadatav2.csv"))
metadata_mh.index = [str(m) for m in metadata_mh["Specimen ID"]]
metadata_mh = metadata_mh.loc[mh_cells]

print(f"✓ Loaded {len(mh_cells)} cells")
print(metadata_mh["Dataset"].value_counts())


# %% [markdown]
# ### 2.4 — UMAP coloured by dataset and sub-class

# %%
reducer_mh = umap.UMAP(metric="precomputed", random_state=1)
embedding_mh = reducer_mh.fit_transform(gw_dist_mh)

px.scatter(
    x=embedding_mh[:, 0], y=embedding_mh[:, 1],
    template="simple_white",
    hover_name=[c + ".swc" for c in metadata_mh.index],
    color=metadata_mh["Dataset"],
    color_discrete_map={"Human Patchseq": "red", "Old Gouwen": "cyan", "New Patchseq": "blue"},
    title="Mouse + Human — Dataset labels",
    width=800, height=600,
)

# %%
px.scatter(
    x=embedding_mh[:, 0], y=embedding_mh[:, 1],
    template="simple_white",
    hover_name=[c + ".swc" for c in mh_cells],
    color=metadata_mh["T type sub-class"],
    title="Mouse + Human — T type sub-class",
    width=800, height=600,
)


# %% [markdown]
# ### 2.5 — Leiden clustering

# %%
clusters_mh = cajal.utilities.leiden_clustering(gw_dist_mh, seed=1)

px.scatter(
    x=embedding_mh[:, 0], y=embedding_mh[:, 1],
    template="simple_white",
    hover_name=[c + ".swc" for c in mh_cells],
    color=[str(m) for m in clusters_mh],
    title="Mouse + Human — Leiden clusters",
    width=800, height=600,
)


# %% [markdown]
# ### 2.6 — Laplacian score

# %%
mh_layers = np.unique(metadata_mh["T type sub-class"])
mh_indicator = (np.array(metadata_mh["T type sub-class"])[:, None] == mh_layers) * 1

laplacian_mh = pd.DataFrame(
    cajal.laplacian_score.laplacian_scores(
        mh_indicator,
        gw_dist_mh,
        np.median(squareform(gw_dist_mh)),
        permutations=5000,
        covariates=None,
        return_random_laplacians=False,
    )[0]
)
laplacian_mh.index = mh_layers
print(laplacian_mh)


# %% [markdown]
# ### 2.7 — Within-dataset CV (all cells pooled)

# %%
cell_type_mh = np.array(metadata_mh["T type sub-class"])

clf_mh = RandomForestClassifier(n_estimators=100, random_state=1)
cv_mh  = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
acc_mh = cross_val_score(clf_mh, X=gw_dist_mh, y=cell_type_mh, cv=cv_mh)
print(f"Within-dataset CV accuracy (sub-class): {np.mean(acc_mh):.4f}")

# %%
cell_type_lamp5 = np.array(metadata_mh["Binary_Lamp5"])
acc_lamp5 = cross_val_score(clf_mh, X=gw_dist_mh, y=cell_type_lamp5, cv=cv_mh)
print(f"Within-dataset CV accuracy (Binary Lamp5): {np.mean(acc_lamp5):.4f}")


# %% [markdown]
# ### 2.8 — Cross-species transfer: train on mouse, test on human
#
# **Fix applied:** For cross-species transfer the test feature matrix must be
# `X[human_rows, mouse_cols]` — i.e. each human cell's distances to every
# mouse training cell. The original code used `X[human_rows, :]` (all columns),
# so the feature dimensionality and semantics were wrong.

# %%
def cross_species_rf(gw, meta, label_col, dataset_col,
                     mouse_labels=("Old Gouwen", "New Patchseq"),
                     human_label="Human Patchseq",
                     n_estimators=100, random_state=0):
    """
    Train RF on mouse cells, predict on human cells.
    Returns (accuracy, y_human, y_pred, subclass_names).
    """
    mouse_mask = meta[dataset_col].isin(mouse_labels).values
    human_mask = (meta[dataset_col] == human_label).values

    y = np.array(meta[label_col])

    # FIX: correct asymmetric slicing for cross-species transfer
    X_train = gw[np.ix_(mouse_mask, mouse_mask)]   # mouse × mouse
    X_test  = gw[np.ix_(human_mask, mouse_mask)]   # human × mouse

    y_train = y[mouse_mask]
    y_test  = y[human_mask]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc, y_test, y_pred, np.unique(y_test), clf


# Sub-class transfer
acc, y_h, y_pred, subclass_names, rf_model = cross_species_rf(
    gw_dist_mh, metadata_mh, "T type sub-class", "Dataset"
)
print(f"Cross-species accuracy (sub-class): {acc:.4f}")

conf = confusion_matrix(y_h, y_pred, labels=subclass_names)
conf_pct = conf / conf.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
sns.set(font_scale=1.4)
sns.heatmap(conf_pct, annot=True, cmap="Blues", fmt=".2f",
            xticklabels=subclass_names, yticklabels=subclass_names, ax=ax)
ax.set_title("Mouse → Human RF (CAJAL) — Sub-class")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "crossspecies_subclass_confusion.png"), dpi=150)
plt.show()


# %% [markdown]
# ### 2.9 — LAMP5 binary cross-species classifier

# %%
acc_l5, y_h_l5, y_pred_l5, _, _ = cross_species_rf(
    gw_dist_mh, metadata_mh, "Binary_Lamp5", "Dataset"
)
print(f"Cross-species accuracy (Binary Lamp5): {acc_l5:.4f}")

subclass_names_l5 = ["Non-Lamp5", "Lamp5"]
conf_l5 = confusion_matrix(y_h_l5, y_pred_l5)
conf_l5_pct = conf_l5 / conf_l5.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
sns.set(font_scale=1.4)
sns.heatmap(conf_l5_pct, annot=True, cmap="Blues", fmt=".2f",
            xticklabels=subclass_names_l5, yticklabels=subclass_names_l5, ax=ax)
ax.set_title("LAMP5 Binary Classifier (Mouse → Human)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "crossspecies_lamp5_binary_confusion.png"), dpi=150)
plt.show()


# %% [markdown]
# ### 2.10 — Permutation null baseline
#
# **Fix applied:** The original code shuffled *test* labels but kept predictions
# identical — not a valid null. A proper null re-trains the classifier on
# randomly shuffled *training* labels N times and reports the distribution of
# null accuracies.

# %%
N_PERMUTATIONS = 100

null_accs = []
mouse_mask = metadata_mh["Dataset"].isin(["Old Gouwen", "New Patchseq"]).values
human_mask = (metadata_mh["Dataset"] == "Human Patchseq").values
y_all      = np.array(metadata_mh["T type sub-class"])

X_train_real = gw_dist_mh[np.ix_(mouse_mask, mouse_mask)]
X_test_real  = gw_dist_mh[np.ix_(human_mask, mouse_mask)]
y_train_real = y_all[mouse_mask]
y_test_real  = y_all[human_mask]

for i in range(N_PERMUTATIONS):
    y_train_shuffled = sk_shuffle(y_train_real, random_state=i)
    rf_null = RandomForestClassifier(n_estimators=100, random_state=i)
    rf_null.fit(X_train_real, y_train_shuffled)
    null_accs.append(accuracy_score(y_test_real, rf_null.predict(X_test_real)))

real_acc = accuracy_score(y_test_real,
                          rf_model.predict(X_test_real))  # rf_model from cell 2.8

print(f"Real accuracy       : {real_acc:.4f}")
print(f"Null mean ± std     : {np.mean(null_accs):.4f} ± {np.std(null_accs):.4f}")
print(f"p-value (empirical) : {np.mean(np.array(null_accs) >= real_acc):.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_accs, bins=20, color="steelblue", edgecolor="white", alpha=0.8, label="Null")
ax.axvline(real_acc, color="red", linewidth=2, label=f"Real ({real_acc:.3f})")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Count")
ax.set_title("Permutation null distribution — Cross-species sub-class")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "permutation_null_distribution.png"), dpi=150)
plt.show()


# %% [markdown]
# ### 2.11 — Original vs shuffled confusion matrices (for visualisation)

# %%
# Original
conf_orig = confusion_matrix(y_test_real,
                             rf_model.predict(X_test_real),
                             labels=subclass_names)
conf_orig_pct = conf_orig / conf_orig.sum(axis=1, keepdims=True)

# Single shuffle (for visual comparison only — use section 2.10 for statistics)
y_single_shuffle = sk_shuffle(y_test_real, random_state=42)
conf_shuf = confusion_matrix(y_single_shuffle,
                             rf_model.predict(X_test_real),
                             labels=subclass_names)
conf_shuf_pct = conf_shuf / conf_shuf.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, mat, title in zip(
    axes,
    [conf_orig_pct, conf_shuf_pct],
    ["Original labels", "Shuffled labels (single draw)"],
):
    sns.heatmap(mat, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=subclass_names, yticklabels=subclass_names, ax=ax)
    ax.set_title(f"Mouse → Human RF — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "crossspecies_original_vs_shuffled.png"), dpi=150)
plt.show()
