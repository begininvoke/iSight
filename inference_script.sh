#!/bin/bash
# Image-level inference with iSight-slide on the validation set.
# Checkpoint: https://huggingface.co/zhihuanglab/iSight-slide  (checkpoints/iSight-slide.pth)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-checkpoints/iSight-slide.pth}
CONFIG_FILE=isight_slide/config/config.ini
INFERENCE_DATA=validation_data/validation_metadata.csv
HDF5_BASE_DIR=validation_data/rle_masks
RLE_MAP_PATH=validation_data/rle_masks/rle_mask_index.json
OUTPUT_DIR=${OUTPUT_DIR:-./results}

cd isight_slide
python scripts/inference.py \
  --config "../$CONFIG_FILE" \
  --inference_data_path "../$INFERENCE_DATA" \
  --checkpoint_path "../$CHECKPOINT_PATH" \
  --hdf5_base_dir "../$HDF5_BASE_DIR" \
  --rle_map_path "../$RLE_MAP_PATH" \
  --output_dir "../$OUTPUT_DIR" \
  --batch_size 4 --num_workers 4 \
  --generate_visualizations --save_name validation_results --save_logits
