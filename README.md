# iSight: Towards expert-AI co-assessment for improved immunohistochemistry staining interpretation
[![arXiv](https://img.shields.io/badge/arXiv-2602.04063-b31b1b.svg)](https://arxiv.org/abs/2602.04063)

Code for **iSight-slide** (image-level IHC interpretation) and **iSight-cell** (cell-level
staining prediction).

## 📊 System overview

![System Architecture](figure/figure_1.png)

**Figure 1: Overview of the HPA10M dataset composition and iSight's multi-task architecture for automated IHC assessment.**

**a**, Workflow illustrating how the dataset is curated from the Human Protein Atlas repository.
**b**, Composition showing 45 distinct normal tissue types included in the HPA10M dataset.
**c**, Breakdown of 20 primary cancer types represented in HPA10M.
**d**, Architecture overview: whole slide images are first partitioned into 336×336 pixel patches, then encoded using CLIP-ViT-large-patch-14-336. Features from all patches are aggregated through CLAM's gated attention mechanism. In parallel, clinical metadata (tissue type, SNOMED codes, antibody details) is processed via CLIP's text encoder.
**e**, Multi-task prediction framework: five independent classification heads process the aggregated features to predict staining characteristics (subcellular location, intensity level, positive cell percentage) and specimen properties (tissue category, malignancy).

*Source: Human Protein Atlas database ([v23.proteinatlas.org](http://v23.proteinatlas.org/ENSG00000170312-CDK1/))*

## 🔗 Links

| Resource | Link |
|----------|------|
| **Training dataset** | [nirschl-lab/hpa10m](https://huggingface.co/datasets/nirschl-lab/hpa10m) |
| **iSight-slide** (checkpoint + code) | [zhihuanglab/iSight-slide](https://huggingface.co/zhihuanglab/iSight-slide) |
| **iSight-cell** (staining model) | [zhihuanglab/iSight-cell](https://huggingface.co/zhihuanglab/iSight-cell) |
| **iSight-target** (target-cell classifier) | [zhihuanglab/iSight-target](https://huggingface.co/zhihuanglab/iSight-target) |

## 📁 Layout

```
isight_slide/
  model/patch_encoder_with_clam.py   CLIP encoder, all-token gated attention,
                                     text (context) branch, cell-type conditioning,
                                     5 multi-task heads
  dataset/hpadataset.py              HPA10M MIL dataset, tissue-mask patching
  main_v2_dist_resume.py             DDP training with resume
  config/config.ini                  the released configuration
  scripts/inference.py               image-level inference
  scripts/infer_rs200_text.py        reader-study inference

isight_cell/code/
  pipeline/                          weakly supervised training, self-refinement,
                                     cell-level staining prediction, evaluation
  pipeline/earlystop.py              val macro-F1 early stopping (min_delta, patience,
                                     keep-current) used to select the released epoch
  target_cell/extract_target_crops.py  crops for the 43 target classes
  target_cell/train_target_head.py     43 binary target-cell heads on UNI2-h
  deps/                              model class and UNI2-h backbone
  tissue.py                          tissue mask
isight_cell/meta/classes_43.csv      the 43 target classes (tissue x cell-type)

validation_data/                     2,000-image validation set (images, RLE masks, metadata)
```

## 🧠 Models

**iSight-slide.** CLIP ViT-L/14-336 patch encoder over all 336 px tissue patches of an image.
Every patch contributes all 576 of its ViT tokens; a gated attention module scores each token
position and softmaxes **across patches** at that position, so pooling is per-token rather than
per-patch. The pooled representation is the mean over tokens. Two conditioning signals are
added to it: a text (context) branch encoding the query (tissue, diagnosis and gene), applied
with dropout during training, and a cell-type embedding. Five linear heads predict:

| Task | Classes | Labels |
|------|---------|--------|
| **Staining intensity** | 4 | negative, weak, moderate, strong |
| **Staining location** | 4 | none, cytoplasmic/membranous, nuclear, cytoplasmic/membranous,nuclear |
| **Staining quantity** | 4 | none, <25%, 25%-75%, >75% |
| **Tissue type** | 58 | human tissue types |
| **Malignancy** | 2 | normal, cancer |

**iSight-cell.** Cellpose-SAM segmentation, then a target-cell classifier selects the cells of
interest, then a UNI2-h backbone fully fine-tuned with dual heads for per-cell staining
intensity (4) and subcellular location (4). Training is weakly supervised from image-level HPA
labels, with a self-refinement round.

The target-cell classifier has one binary head per class in `isight_cell/meta/classes_43.csv`,
43 tissue × cell-type classes. That table is the single definition of the class space: crop
extraction applies it, so `class_idx` is written in the 0..42 space, and the trainer and the
selector both use it directly with no run-time conversion.

## 🚀 Setup

**Typical install time**: ~5–10 minutes (depending on network speed for PyTorch and model weights).

```bash
conda create -n isight python=3.10 -y && conda activate isight
# PyTorch with CUDA, matching your driver (nvidia-smi); versions at pytorch.org/get-started/previous-versions
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Checkpoint:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("zhihuanglab/iSight-slide", "checkpoints/iSight_model_checkpoint.pth")
```

## ▶️ Running iSight-slide

**Inference** on the validation set:

```bash
CHECKPOINT_PATH=/path/to/iSight_model_checkpoint.pth bash inference_script.sh
```

**Expected run time**: on a standard GPU each image is processed within a few seconds.
Outputs in `results/`:

| File | Description |
|------|-------------|
| `inference_results_<save_name>.csv` | Per-sample predictions with ground truth, predicted labels and logits for all 5 tasks |
| `metrics_<save_name>.csv` / `.json` | Accuracy, balanced accuracy and weighted F1 per task |
| `inference_config.json` | Configuration used for the run |
| `visualizations/<save_name>/` | Confusion matrices per task (with `--generate_visualizations`) |

**Training**:

```bash
export ISIGHT_DATA_ROOT=/path/to/hpa10m          # metadata, RLE masks, images
cd isight_slide && python main_v2_dist_resume.py --config config/config.ini
```

Data locations are environment variables, not hard-coded paths:

| variable | what |
|---|---|
| `ISIGHT_DATA_ROOT` | root for the defaults below |
| `ISIGHT_TRAIN_META` / `ISIGHT_TEST_META` | HPA10M split metadata (feather) |
| `ISIGHT_RLE_DIR` / `ISIGHT_RLE_INDEX` | RLE tissue masks and their index |
| `ISIGHT_IMAGE_DIR` | images, only for the `simple_downsample` version |
| `SCHEDULER_PER_EPOCH=1` | step the LR scheduler per epoch instead of per batch |

The released configuration uses `batch_size = 1`, which is what the checkpoint was trained with.

## ▶️ iSight-cell pipeline, in run order

| # | script | what it does |
|---|---|---|
| 1 | `pipeline/tissue_mask_gen.py` | tissue mask per image (`tissue.py` holds the pixel rule) |
| 2 | `target_cell/extract_target_crops.py` | 64x64 crops for the 43 target classes; writes `class_idx` as 0..42 |
| 3 | `target_cell/train_target_head.py` | fine-tunes UNI2-h with 43 binary target-cell heads |
| 4 | `pipeline/select_target_v2.py` | applies that classifier per image, keeps the target cells |
| 5 | `pipeline/train_foundation.py` | staining model, step 1: broadcast image labels onto cells |
| 6 | `pipeline/predict_all_target.py` | scores every cell with the step-1 model |
| 7 | `pipeline/refine_target.py` | keeps cells whose prediction agrees with the image label on both heads |
| 8 | `pipeline/scan_refined.py` | per-cell index over the refined pool (`$SCAN_NPZ` for step 2) |
| 9 | `pipeline/train_shards_resample.py` | staining model, step 2: balanced resampling; **this produced the released checkpoint** |

Evaluation: `pipeline/val_richeval.py` (validation, image-level accuracy and QWK),
`pipeline/eval_test500k_fixedloc.py` (held-out 500K set), `pipeline/eval_flats_fixed.py`
(any image list, same fixed aggregation), `pipeline/agg_uncap.py` (image-level metrics with no
per-image cell cap). Segmentation upstream of step 1 is Cellpose-SAM at its released settings.

| variable | what |
|---|---|
| `ISIGHT_ROOT` | project root for the iSight-cell scripts |
| `UNI2_CKPT_PATH` | UNI2-h weights ([MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)) |
| `EARLY_STOP` / `ES_METRIC` / `ES_MIN_DELTA` / `ES_PATIENCE` | early stopping; defaults are the released settings (on, `val_cell_avg_f1`, 0.001, 2) |

## 📦 Checkpoints

| model | config | note |
|---|---|---|
| iSight-slide | `isight_slide/config/config.ini` — `v3_all_tokens`, batch_size 1, lr 1e-6, 10 epochs | [zhihuanglab/iSight-slide](https://huggingface.co/zhihuanglab/iSight-slide) |
| iSight-cell — staining | step 2 (balanced resampling), lr 2e-5, **epoch 9** | early stopping on validation macro-F1 (min_delta 0.1%, patience 2, keep-current) |
| iSight-cell — target selection | 43 binary heads, `isight_cell/meta/classes_43.csv` | val mean F1 0.9954 (ep4); pass it as `CKPT` |

## 📄 License

See [LICENSE](LICENSE) (PENN Academic Software License Agreement).

## 📧 Contact

Zhi Huang — [zhi.huang@pennmedicine.upenn.edu](mailto:zhi.huang@pennmedicine.upenn.edu)
