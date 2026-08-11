# ORBIT

[![ORBIT version](https://img.shields.io/badge/ORBIT-1.2.0-6f42c1.svg)](https://github.com/M-Fotheringham/ORBIT/releases)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)

## Operator-guided Random-field Biomarker Immunophenotyping Training

![ORBIT logo](docs/figs/main_page_logo.png)

ORBIT is an interactive desktop application for supervised phenotyping of multiplex immunofluorescence images. It combines CellPoseSAM cell segmentation and machine-learning cell classification in one pathologist-friendly workflow.

Review segmentation quality, training labels, and phenotype calls before using exported results in downstream analyses.

## Highlights

- Load multiple TIFF, QPTIFF, or local OME-Zarr images into one project.
- Navigate 512 × 512-pixel fields of view (FOVs) and jump anywhere in the source image with the clickable whole-image overview.
- Import existing cell-level measurements and label masks, or generate both with CellPoseSAM.
- Train phenotypes manually with a random forest model or by fluorescence thresholds, or have ORBIT automatically assign cell labels that can be reviewed and edited.
- Save projects, import/export reusable phenotype models, export generated segmentation, and export cell-level phenotype tables.

## ORBIT acronym

**O**perator-guided  
**R**andom-field  
**B**iomarker  
**I**mmunophenotyping  
**T**raining

## Requirements

### Running ORBIT

- Python 3.11
- Astral's uv package manager

### Running CellPoseSAM segmentation

CellPoseSAM segmentation requires an NVIDIA GPU that PyTorch can access through CUDA. ORBIT reports GPU availability at the top of the **Segmenting > CellPoseSAM** panel and disables the segmentation controls when a compatible GPU is not detected. It would otherwise run for hours, if given a whole-slide image.

An AMD GPU is not CUDA-compatible; however, you can still load images, import existing segmentation, navigate FOVs, phenotype cells, and export results without a CUDA GPU.

## Installation

### Windows standalone application

Download the latest installer for Windows from [Windows Installer (Releases)](https://github.com/M-Fotheringham/ORBIT/releases).

### Editable source installation with uv

ORBIT 1.2.0 uses [uv](https://docs.astral.sh/uv/) and a `uv.lock` file to manage its environment. You do not need to create or activate a conda environment.

1. Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

   On Windows, uv can be installed from PowerShell with WinGet:

   ```powershell
   winget install --id=astral-sh.uv -e
   ```

2. Clone and enter the repository:

   ```powershell
   git clone https://github.com/M-Fotheringham/ORBIT.git
   cd ORBIT
   ```

3. Create/synchronize the project environment from the lockfile:

   ```powershell
   uv sync --locked
   ```

4. Start ORBIT:

   ```powershell
   uv run orbit
   ```

### Updating a source checkout

```powershell
git pull
uv sync --locked
uv run orbit
```

## Supported data

### Images

ORBIT can add:

- `.qptiff`, `.tif`, and `.tiff` files through **TIFF / QPTIFF**;
- a local OME-Zarr directory through **OME-Zarr directory**.

Channel names are read from image metadata when available. A channel whose name contains `DAPI` is used for the nuclear overlay, DAPI-positive FOV selection, and the CellPoseSAM nuclear input. For OME-Zarr data with time or Z dimensions, ORBIT currently displays the first time point and first Z plane.

### Existing segmentation

Segmentation is optional when an image is first added. To import it later, select the image in the carousel and click **Load Segmentation**. ORBIT then asks for:

1. a non-empty cell-data table (`.tsv`, `.txt`, or `.csv`); and
2. a two-dimensional labelled mask (`.tif` or `.tiff`) with the same width and height as the image.

ORBIT was designed to read [CellPose](https://github.com/mouseland/cellpose) segmentation data.

## Quick start

1. Click **Add Image**, choose **TIFF / QPTIFF** or **OME-Zarr directory**, and select the image.
2. Add more images if required; use the bottom carousel to select the active image.
3. For each image, either click **Load Segmentation** to import a cell table and mask, or run **Segmenting > CellPoseSAM** once the project images are loaded.
4. Click **Generate FOV** and choose the fluorescence marker and pseudocolour to display.
5. Choose **Phenotyping > Automated**, **Random Forest**, or **Threshold Slider**.
6. Review calls, correct labels where appropriate, and apply the result to all loaded images.
7. Save the project and export the model and/or cell phenotype table.


# User Guide

## 1. Loading

### 1.a Adding images to a project

1. Click 'Add Image'
2. Select a TIFF/QPTIFF image or OME-Zarr directory containing the fluorescence data
3. Optionally click 'Load Segmentation' to select an existing cell-data TSV and segmentation-mask TIFF. Segmentation is not required when adding an image.

### 1.b Opening a pre-existing project

1. File > Open...
2. Select the .orbit.json file corresponding to your project

### 1.c Importing a pre-existing phenotyping model

1. File > Import Model...
2. Select the .orbitmodel file corresponding to your model

## 2. Navigating

### 2.a Generating fields of view (FOV)

1. Click 'Generate FOV'. This will produce a random 512x512 px field of view
2. To generate another random field, click 'Generate FOV' again

### 2.b Changing the displayed marker channel and colour

1. Use the marker dropdown list to select a channel
2. Use the colour dropdown list to select a pseudo colour for the marker

### 2.c Toggling DAPI and segmentation overlays

The DAPI stain and the segmentation can be toggled on/off.

### 2.d Cycling through images

Change the image displayed using the carousel arrows or by selecting an image from the dropdown list.

### 2.e Segmenting with CellPoseSAM

1. Select Segmenting > CellPoseSAM and choose one or more membrane-guiding markers. Selected channels are normalized and merged, and DAPI is supplied as nuclear guidance when available.
2. Click 'Preview Current FOV (CPU)' to segment only the displayed field without modifying project data. Review the overlay, then click 'Accept Preview' or 'Discard'.
3. On systems with a CUDA-compatible GPU, 'Segment All Images (GPU)' replaces the segmentation and cell-level measurements for every loaded image.
4. Click 'Export Segmentation' to save generated cell-data TSV and mask TIFF files.

## 3. Phenotyping

### 3.a Adding and removing phenotype training labels

1. Provide a name for the phenotype algorithm being trained
2. Click on a cell to label it positive or negative for your desired phenotype. You can remove the cell from training by clicking it and selecting 'Do not train'.

### 3.b Training machine-learning phenotyping model

Choose 'Select Features...' to control which shared numeric cell measurements are used. Once positive and negative labels have been selected, click 'Train Model' to train a random forest classifier. Use the 'Probability positive' slider (default 50%) to adjust the positive-call decision threshold.

### 3.c Quality-checking model performance and training labels

1. Cycle through labeled cells using the arrows in the 'Phenotype Training' box. This will centre the image on trained cells.
2. Once a model has been trained, click 'Apply to Loaded Images' to see the performance of the model on untrained cells.

### 3.d Threshold-slider phenotyping

Select Phenotyping > Threshold Slider to set a fluorescence-intensity threshold and the percentage of positive pixels required to call a cell positive. The fluorescence histogram displays `log2(1 + mean intensity)` across all cells in the current image; the second histogram displays positive-pixel percentages. Compartment checkboxes and the inward-buffer slider control which cell pixels form the denominator.

### 3.e Automated phenotyping

Select Phenotyping > Automated and click 'Auto Phenotype' to reproducibly generate training labels, train a random forest, and apply it to all loaded images. Feature selection and the positive-probability threshold can be adjusted before training. Click 'Edit' to review labels and threshold settings, then 'Re-Phenotype' to apply changes.

## 4. Saving and Exporting

### 4.a Saving a project

1. Click File > Save or Save As... to save a project. Asset paths are stored relative to the project file whenever possible, allowing the project folder to be moved as long as its internal file structure is preserved.

### 4.b Saving a phenotyping model

1. Click File > Export Model... to save a phenotyping model.

### 4.c Exporting phenotyping data

1. Once a model has been trained and applied to loaded images, click 'Export Cell Phenotypes' to save every original cell-data column, phenotype labels, positive probabilities, image names, and relevant thresholds for all loaded images.
2. ORBIT also writes a companion `.provenance.json` file recording the ORBIT version, method, model settings, selected features, source paths, and segmentation metadata.

## 5. Batch/headless model application

An exported model can be applied to every cell-data table referenced by a portable project without opening the GUI:

```bash
uv run orbit-batch --model phenotype.orbitmodel --project project.orbit.json --output phenotypes.tsv
```

Use `--probability-threshold 0.60` to override the decision threshold stored in the model. The batch export includes the complete input tables, phenotype labels, positive probabilities, and a companion provenance JSON file.

## Development

Create a branch before making changes:

```powershell
git checkout -b feature/your-feature
```

Synchronize the locked environment and run the application from the checkout:

```powershell
uv sync --locked
uv run orbit
```

If dependencies in `pyproject.toml` change, refresh and verify the lockfile:

```powershell
uv lock
uv sync
```

Build the Python source distribution and wheel with:

```powershell
uv build
```

`uv build` writes Python package artifacts to `dist/`; it does not create a standalone Windows executable or installer.

Commit and push the branch for review:

```powershell
git add .
git commit -m "Describe the change"
git push -u origin feature/your-feature
```

## Troubleshooting

### The CellPoseSAM panel is disabled

ORBIT could not detect an NVIDIA CUDA GPU through PyTorch. Confirm that the NVIDIA driver is installed, then start ORBIT from the same uv environment in which its CUDA-enabled PyTorch build is installed. A CPU or AMD-only system can use imported segmentation but cannot run the current CellPoseSAM workflow.

### A project no longer opens its data

The project stores absolute paths. Restore the images, OME-Zarr directories, cell tables, and masks to their saved locations, or add them again and save a new project.

### A cell table cannot be mapped to the mask

Confirm that the mask is two-dimensional and matches the image dimensions. The table should include numeric cell IDs matching the positive mask labels, or valid X/Y centroid coordinates located inside or close to each segmented cell.

### Random FOV generation cannot find tissue

Confirm that the nuclear channel name contains `DAPI`, that the channel is not blank, and that the image contains fields with at least 1% DAPI-positive pixels.


## Questions and bug reports

Open an issue in the [GitHub issue tracker](https://github.com/M-Fotheringham/ORBIT/issues) and include the ORBIT version, operating system, image format, steps to reproduce, and the full error message or traceback.
