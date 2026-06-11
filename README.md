# 🧬 Cross-Species Neuron Morphology-Based Cell-Type Classification

**Nandita Jayadevan**  
Supervisor: Dr. Shreejoy Tripathy  
HMB496 Undergraduate Thesis Project — University of Toronto (2024)

---

## 📌 Overview

This project investigates whether neuronal morphology alone can be used to classify interneuron cell types across species (mouse → human) using machine learning applied to 3D neuron reconstructions.

We leverage publicly available Patch-seq and morphological datasets from the Allen Institute to build cross-species classification models and evaluate the conservation of interneuron morphology across species.

---

## 🎯 Research Question

Can morphological features of neurons reliably predict conserved interneuron subclasses across mouse and human brains?

---

## 🧪 Datasets

This study uses three main datasets:

- Mouse visual cortex interneurons (Gouwens et al., 2019)
- Mouse Patch-seq interneurons (Gouwens et al., 2020)
- Human interneuron reconstructions (Lee et al., 2023)

All neurons are represented in **SWC format** with matched transcriptomic cell-type labels.

---

## ⚙️ Methods

### 🧬 Feature Extraction

Two complementary morphology featurization approaches were used:

- **CAJAL**  
  Generates Gromov–Wasserstein distance-based embeddings of neuronal morphologies.

- **Skeleton-Keys (Allen Institute)**  
  Extracts layer-aligned dendritic and axonal morphological features.

---

### 🤖 Machine Learning Models

- Random Forest Classifier
- Dummy baseline classifier (random label permutation)
- Stratified 5-fold cross-validation (mouse datasets)
- Train–test cross-species evaluation (mouse → human)

---

### 📊 Evaluation Metrics

- Accuracy (primary metric)
- Confusion matrices
- UMAP visualization of learned feature space
- Cross-species generalization performance

---

## 📊 Key Results

- **Mouse morphology classification accuracy:** ~61%
- **Human morphology classification accuracy:** ~35%
- **Cross-species transfer (mouse → human):** ~28%
- **LAMP5 subclass transfer accuracy:** ~67% (notable conservation signal)

---

## 🔬 Main Findings

- Morphological features partially encode interneuron identity within species.
- Strong performance in mouse datasets suggests robust morphological signatures.
- Human interneurons exhibit higher morphological variability and reduced classification accuracy.
- LAMP5 interneurons show consistent cross-species morphological conservation.
- Axonal features introduce noise; dendrite + soma features are more stable for classification.

---

## 📁 Repository Structure
