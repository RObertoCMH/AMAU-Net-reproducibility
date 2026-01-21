# AMAU-Net reproducibility (F3 Netherlands)

This repository provides configuration files and code to reproduce a controlled ablation study for section-based seismic facies segmentation on the F3 Netherlands benchmark.

We evaluate three closely related configurations under identical training conditions:

- **U-Net (amplitude-only)**: 1 input channel, no attention  
- **U-Net + instantaneous attributes**: 4 input channels (amplitude + 3 attributes), no attention  
- **AMAU-Net**: 4 input channels + selected CBAM insertions + bottleneck self-attention  

> **Note on naming:** the implementation uses a single model class (`UNetWithAttention`).  
> AMAU-Net corresponds to a specific configuration of that class (multi-attribute input + attention toggles).

---

## Repository structure

- `configs/`  
  YAML configuration files for split paths, attribute definition, model variants, training, and evaluation.

- `data_instructions/`  
  Instructions to obtain the F3 dataset from official sources and to arrange it locally (data is not redistributed here).

- `src/amaunet/`  
  Reusable Python modules (attributes, model definition, utilities).

- `scripts/`  
  Entry-point scripts to run the pipeline end-to-end (attribute computation → training → evaluation).

- `paper_figures/`  
  Final figures used in the manuscript (PDF/PNG).

---

## Data availability (important)

This repository **does not redistribute** the F3 dataset or any derived large arrays.  
You must download the dataset from official/public sources and store it locally.

See: `data_instructions/GET_F3.md`.

### Expected local folder layout (current)
The current setup assumes **precomputed splits** stored as NumPy arrays:

- `train/train_seismic.npy` and `train/train_labels.npy`
- `validation/test2_seismic.npy` and `validation/test2_labels.npy`
- `test/test1_seismic.npy` and `test/test1_labels.npy` (Benchmark Test set #1)

Set the base directory in `configs/split.yaml`:
- `data.base_dir: "PATH/TO/YOUR/data"`

---

## Instantaneous attributes (exact repo convention)

The 4-channel input is built as:

1. amplitude  
2. instantaneous phase (wrapped, radians)  
3. instantaneous frequency (rad/sample)  
4. envelope  

Attribute computation follows the current implementation:

- analytic signal (trace-wise along the time axis): `hilbert(x)`
- envelope: `abs(analytic_signal)`
- phase: `angle(analytic_signal)` (wrapped)
- angular instantaneous frequency: `gradient(unwrap(phase))`  
  → **units: rad/sample** (no division by `2π dt`)

See: `configs/attributes.yaml`.

---

## Model variants (exact manuscript setting)

CBAM insertions are supported only at the layer IDs implemented in `UNetWithAttention`:

- encoder: `2, 3, 4` (after conv2/conv3/conv4)
- decoder: `6, 7, 8` (after conv6/conv7/conv8)

**Paper setting (AMAU-Net):**
- CBAM enabled at **[2, 4, 8]**
- self-attention enabled at the **bottleneck** (after conv5)

See: `configs/model.yaml`.

---

## Requirements

The code is written in Python and uses PyTorch for training.

Core dependencies (from the original notebooks / planned scripts):
- Python 3.9+ (recommended)
- numpy, scipy
- torch, torchvision
- albumentations
- matplotlib, pandas
- tqdm
- torchmetrics, scikit-image

---

## Environment setup (example)## Environment setup (example)



Using conda:

```bash
conda create -n amau python=3.10 -y
conda activate amau
pip install -r requirements.txt

