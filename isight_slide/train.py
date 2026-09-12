import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_TIMEOUT'] = '120'
import torch
import torch.nn as nn
from dataset.hpadataset import HPADatasetMIL, HPADatasetDownsample, SeededSampler, ResumableDistributedSampler, HPADatasetMIL_url
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import configparser
import argparse
import wandb
import copy
from sklearn.model_selection import train_test_split
from datetime import datetime
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.cuda.amp  # Import Automatic Mixed Precision
import re
import random

"""
Be sure to init conda env:
conda activate hpa

Note: If running on a non-H100 GPU, there might be an error for pyarrow.
Reinstall pyarrow with following command will resolve the issue:
conda install -c conda-forge pyarrow
"""

"""
CrossEntropyLoss is better than BCEWithLogitsLoss for multi-class classification
because it directly handles mutually exclusive classes by computing probabilities
over all classes, whereas BCEWithLogitsLoss treats each class independently.

If use BCEWithLogitsLoss for multi-class classification, it treats each class as
independent and computes a separate binary classification for each class. This is
problematic because, in multi-class classification, only one class can be correct
for each input, and the classes are mutually exclusive. BCEWithLogitsLoss doesn't
enforce this exclusivity, leading to incorrect probability distributions where
multiple classes could be predicted as "active" at once, which is not appropriate
for multi-class problems where exactly one class should be correct.
"""
criterion_intensity = nn.CrossEntropyLoss()  # For intensity
criterion_location = nn.CrossEntropyLoss()   # For location
criterion_quantity = nn.CrossEntropyLoss()   # For quantity
criterion_tissue = nn.CrossEntropyLoss()     # For tissue type (58 classes)
criterion_malignancy = nn.CrossEntropyLoss() # For tumor vs non-tumor (binary)

# ---------------------------------------------------------------- data locations
# HPA10M metadata and RLE tissue masks (environment variables; see README).
DATA_ROOT = os.environ.get("ISIGHT_DATA_ROOT", "")
DATA_TRAIN_META = os.environ.get("ISIGHT_TRAIN_META",
                                 os.path.join(DATA_ROOT, "remaining_training_metadata.feather"))
DATA_TEST_META = os.environ.get("ISIGHT_TEST_META",
                                os.path.join(DATA_ROOT, "validation_metadata.feather"))
DATA_RLE_DIR = os.environ.get("ISIGHT_RLE_DIR", os.path.join(DATA_ROOT, "rle_masks"))
# only used by the `simple_downsample` model version
DATA_IMAGE_DIR = os.environ.get("ISIGHT_IMAGE_DIR", os.path.join(DATA_ROOT, "hpa10m"))
DATA_RLE_INDEX = os.environ.get("ISIGHT_RLE_INDEX",
                                os.path.join(DATA_ROOT, "rle_mask_index.json"))
# Step ReduceLROnPlateau once per epoch instead of once per batch (default: per batch).
SCHEDULER_PER_EPOCH = os.environ.get("SCHEDULER_PER_EPOCH", "0") == "1"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def custom_collate_fn(batch):
    # If no valid samples, return None
    batch_valid = [batch_item for batch_item in batch if batch_item is not None]
    
    if len(batch_valid) == 0:
        return None
    images, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = zip(*batch_valid)
    # Note: here images is a list of (N, 3, 224, 224), where N is the number of patches per image, and is different for each image.
    
    # Convert to tensors and stack
    staining_intensity = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_intensity])
    staining_location = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_location])
    staining_quantity = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in staining_quantity])
    malignancy = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in malignancy])
    tissue_one_hot = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in tissue_one_hot])
    cell_type_one_hot = torch.stack([torch.tensor(sublist, dtype=torch.float) for sublist in cell_type_one_hot])
    
    return images, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot


def train(config):
    # Setup
    if config.distributed:
        # Set the device and get the rank of the current process
        device = torch.device('cuda', config.local_rank)
        torch.cuda.set_device(device)
        rank = torch.distributed.get_rank()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank = 0  # Default rank for non-distributed training

    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()  # For half-precision scaling

    base_model_name = config.base_model_name

    # Model initialization (unchanged)
    if config.model_version == "v1":
        from model.patch_encoder_with_clam import create_clam_vit
        model, optimizer, scheduler = create_clam_vit(
            base_model_name=base_model_name,
            device=device,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            freeze_vit=config.freeze_vit,
            freeze_query_features_encoder=config.freeze_query_features_encoder,
            use_cell_type_embedding=config.use_cell_type_embedding,
        )
    elif config.model_version == "v2":
        from model.patch_encoder_with_clam_v2 import create_clam_vit
        model, optimizer, scheduler = create_clam_vit(
            base_model_name=base_model_name, 
            device=device,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            freeze_vit=config.freeze_vit,
            freeze_query_features_encoder=config.freeze_query_features_encoder,
            use_cell_type_embedding=config.use_cell_type_embedding
        )
    elif config.model_version == "v3":
        from model.patch_encoder_with_clam_v3 import create_clam_vit
        model, optimizer, scheduler = create_clam_vit(
            base_model_name=base_model_name,
            training_config=config,
            device=device,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            freeze_vit=config.freeze_vit,
            freeze_query_features_encoder=config.freeze_query_features_encoder,
            use_cell_type_embedding=config.use_cell_type_embedding
        )
    elif config.model_version == "v3_all_tokens":
        from model.patch_encoder_with_clam_v3_all_tokens import create_clam_vit
        model, optimizer, scheduler = create_clam_vit(
            base_model_name=base_model_name,
            training_config=config,
            device=device,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            freeze_vit=config.freeze_vit,
            freeze_query_features_encoder=config.freeze_query_features_encoder,
            use_cell_type_embedding=config.use_cell_type_embedding
        )
    elif config.model_version == "simple_downsample":
        from model.simple_downsample import create_vit
        model, optimizer, scheduler = create_vit(
            base_model_name=base_model_name, 
            device=device,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            freeze_vit=config.freeze_vit,
            freeze_query_features_encoder=config.freeze_query_features_encoder,
            use_cell_type_embedding=config.use_cell_type_embedding
        )
    model.to(device)

    # Wrap the model with DistributedDataParallel if using distributed training
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=True
        )

    if base_model_name == "vinid/plip":
        patch_size = 224
    elif base_model_name == "openai/clip-vit-large-patch14-336":
        patch_size = 336
    else:
        raise ValueError(f"Model {base_model_name} not supported.")

    processor = model.module.patch_processor if config.distributed else model.patch_processor

    start_epoch = 0
    start_batch_idx = 0
    num_images_so_far = 0
    shuffle_seed = None

    if config.load_checkpoint:
        print(f"Loading checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        if config.distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract epoch, batch_idx, and shuffle_seed from checkpoint
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx'] + 1  # Start from next batch
        shuffle_seed = checkpoint['shuffle_seed']
        print(f"Resuming training from epoch {start_epoch}, batch {start_batch_idx}, with shuffle_seed {shuffle_seed}")

    else:
        shuffle_seed = random.randint(0, 2**32 - 1)

    # Modify the dataset creation

    # # use 2000 validation for training validation
    # validation_set = hpa10m_index[hpa10m_index['split'] == 'validation']
    # hpa10m_test = validation_set.sample(n=2000, random_state=42)
    # remaining_validation = validation_set.drop(hpa10m_test.index).reset_index(drop=True)

    # # add the rest of val to the training set
    # hpa10m_train = pd.concat([hpa10m_index[hpa10m_index['split'] == 'train'], remaining_validation]).reset_index(drop=True)
    # hpa10m_test = hpa10m_test.reset_index(drop=True)

    # data locations: environment variables, see README
    hpa10m_train = pd.read_feather(DATA_TRAIN_META)
    hpa10m_test = pd.read_feather(DATA_TEST_META)
    overlap = pd.merge(hpa10m_train, hpa10m_test, on="name")
    if rank == 0:
        print(f"Overlap: {len(overlap)} (suppose to be 0)")
    # hpa10m_train = hpa10m_train[~hpa10m_train['url'].isin(overlap['url'])]

    if config.use_cell_type != "All":
        hpa10m_train = hpa10m_train.loc[hpa10m_train["cell_type"] == config.use_cell_type, ].reset_index(drop=True)
        # split 95% into train, 5% into validation
        hpa10m_train, hpa10m_test = train_test_split(hpa10m_train, test_size=0.01, random_state=42)
        hpa10m_train = hpa10m_train.reset_index(drop=True)
        hpa10m_test = hpa10m_test.reset_index(drop=True)
        if rank == 0:
            print(f"Using {config.use_cell_type} only. Which subset to {len(hpa10m_train)} images.")
    # unique_tissue_types = [val.lower().replace("cancer","").replace("tissue","").strip() for val in hpa10m_train["tissue"].unique()]
    
    # `datadir` is only read by the `simple_downsample` model version
    datadir = DATA_IMAGE_DIR
    hdf5_base_dir = DATA_RLE_DIR
    rle_map_path = DATA_RLE_INDEX

    if config.model_version == "simple_downsample":
        dataset = HPADatasetDownsample(hpa10m_train, datadir, data_split="train", target_size=patch_size, processor=processor)
    else:
        # dataset = HPADatasetMIL(hpa10m_train, datadir, data_split="train", patch_size=patch_size, processor=processor)
        dataset = HPADatasetMIL_url(hpa10m_train, rle_map_path, hdf5_base_dir, data_split="train", patch_size=patch_size, processor=processor)

    # Calculate the number of batches per GPU
    num_gpus = torch.distributed.get_world_size() if config.distributed else 1
    num_batches_per_gpu = len(dataset) // (config.batch_size * num_gpus)

    # Adjust start_batch_idx if it is too large
    if start_batch_idx >= num_batches_per_gpu:
        print(f"Warning: start_batch_idx ({start_batch_idx}) is larger than available batches per GPU ({num_batches_per_gpu}). Resetting to 0.")
        start_batch_idx = 0

    # Create test dataset and dataloader
    if config.model_version == "simple_downsample":
        test_dataset = HPADatasetDownsample(hpa10m_test, datadir, data_split="test", target_size=patch_size, processor=processor)
    else:
        test_dataset = HPADatasetMIL_url(hpa10m_test, rle_map_path, hdf5_base_dir, data_split="test", patch_size=patch_size, processor=processor)
    
    if config.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_shuffle = False
    else:
        test_sampler = None
        test_shuffle = False

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=test_shuffle,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
        prefetch_factor=8,
        persistent_workers=True,
        sampler=test_sampler
    )

    # Training parameters
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate  # lr = 1e-3 does not reduce the loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_sched_loss = 0.0

        if epoch == start_epoch and config.load_checkpoint:
            current_seed = shuffle_seed
        else:
            current_seed = random.randint(0, 2**32 - 1)
            shuffle_seed = current_seed  # Update the shuffle_seed

        # Set seeds for reproducibility
        set_seed(current_seed + epoch)  # Adding epoch to seed for variety
    
        if config.distributed:
            train_sampler = ResumableDistributedSampler(
                dataset,
                num_replicas=num_gpus,
                rank=rank,
                shuffle=config.shuffle_training,
                seed=current_seed
            )
            train_sampler.set_epoch(epoch)
            if epoch == start_epoch and start_batch_idx > 0:
                skip_samples = start_batch_idx * config.batch_size
                train_sampler.set_start_index(skip_samples)
        else:
            train_sampler = SeededSampler(dataset, shuffle=config.shuffle_training, seed=current_seed)
            if epoch == start_epoch and start_batch_idx > 0:
                skip_samples = start_batch_idx * config.batch_size
                train_sampler.set_start_index(skip_samples)

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=False,
            prefetch_factor=8,
            persistent_workers=True
        )
        num_batches = len(dataloader)
        if epoch == start_epoch:
            num_images_so_far = (epoch * num_batches + start_batch_idx) * config.batch_size
        else:
            num_images_so_far = epoch * num_batches * config.batch_size


        for batch_idx, batch in enumerate(tqdm(dataloader, disable=(rank != 0))):
            # Skip batches if resuming from a specific batch index
            # if epoch == start_epoch and batch_idx < start_batch_idx:
            #     continue

            num_images_so_far += config.batch_size
            if batch is None:
                if rank == 0:
                    print("Batch is None - Maybe invalid image.")
                continue
            patches, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = batch
            # Patches is a tuple of K patches, each patch shape: (N, 3, 224, 224)
            patches = [patch.to(device) for patch in patches]
            # Move tensors to device
            staining_intensity = staining_intensity.to(device)
            staining_location = staining_location.to(device)
            staining_quantity = staining_quantity.to(device)
            tissue_one_hot = tissue_one_hot.to(device)
            malignancy = malignancy.to(device)
            
            # Backward pass and optimization
            optimizer.zero_grad()

            st = time.time()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                # Forward pass
                intensity_out, location_out, quantity_out, tissue_out, malignancy_out, A_raw = model(
                    patches, query_input, cell_type_one_hot, phase="train"
                )
                
                # Compute losses
                intensity_loss = criterion_intensity(intensity_out, torch.argmax(staining_intensity, dim=1))
                location_loss = criterion_location(location_out, torch.argmax(staining_location, dim=1))
                quantity_loss = criterion_quantity(quantity_out, torch.argmax(staining_quantity, dim=1))
                tissue_loss = criterion_tissue(tissue_out, torch.argmax(tissue_one_hot, dim=1))
                malignancy_loss = criterion_malignancy(malignancy_out, torch.argmax(malignancy, dim=1))

                # Combine losses
                total_loss = intensity_loss + location_loss + quantity_loss + tissue_loss + malignancy_loss

            # Backward pass and optimization with scaling
            if config.amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # LR scheduler: step on the loss averaged across ranks so all ranks step together;
            # per batch by default, per epoch with SCHEDULER_PER_EPOCH=1.
            _sched_loss = total_loss.detach()
            if config.distributed:
                torch.distributed.all_reduce(_sched_loss, op=torch.distributed.ReduceOp.SUM)
                _sched_loss = _sched_loss / torch.distributed.get_world_size()
            if not SCHEDULER_PER_EPOCH:
                scheduler.step(_sched_loss.item())
            epoch_sched_loss += _sched_loss.item()

            epoch_loss += total_loss.item()

            et = time.time()
            training_time = et - st

            # Logging and printing only in the main process
            if batch_idx % 100 == 0:
                if rank == 0:
                    # print("="*100)
                    # Print last evaluation performances
                    # tqdm.write("Last Evaluation Performances:")
                    # tqdm.write(f"  Test Loss: {wandb.run.summary.get('test_loss', -1):.4f}")
                    # tqdm.write(f"  Test Intensity Accuracy: {wandb.run.summary.get('test_intensity_accuracy', -1):.4f}")
                    # tqdm.write(f"  Test Location Accuracy: {wandb.run.summary.get('test_location_accuracy', -1):.4f}")
                    # tqdm.write(f"  Test Quantity Accuracy: {wandb.run.summary.get('test_quantity_accuracy', -1):.4f}")
                    # tqdm.write(f"  Test Tissue Accuracy: {wandb.run.summary.get('test_tissue_accuracy', -1):.4f}")
                    # tqdm.write(f"  Test Malignancy Accuracy: {wandb.run.summary.get('test_malignancy_accuracy', -1):.4f}")
                    # tqdm.write("")

                    # # Update tqdm with losses
                    # tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+start_batch_idx+1}/{len(dataloader)}", end="  ")
                    # tqdm.write(f"  Intensity Loss: {intensity_loss.item():.4f}", end="  ")
                    # tqdm.write(f"  Location Loss: {location_loss.item():.4f}", end="  ")
                    # tqdm.write(f"  Quantity Loss: {quantity_loss.item():.4f}", end="  ")
                    # tqdm.write(f"  Tissue Loss: {tissue_loss.item():.4f}", end="  ")
                    # tqdm.write(f"  Malignancy Loss: {malignancy_loss.item():.4f}", end="  ")
                    # tqdm.write(f"  Total Loss: {total_loss.item():.4f}")

                    # print(f"Number of images learned: {len(patches)}, Training: {training_time:.2f} seconds")

                    # Log losses per iteration
                    wandb.log({
                        "epoch": epoch + 1,
                        "batch": batch_idx + start_batch_idx + 1,
                        "intensity_loss": intensity_loss.item(),
                        "location_loss": location_loss.item(),
                        "quantity_loss": quantity_loss.item(),
                        "tissue_loss": tissue_loss.item(),
                        "malignancy_loss": malignancy_loss.item(),
                        "total_loss": total_loss.item(),
                        "training_time": training_time,
                        "num_images_so_far": num_images_so_far,
                        "test_loss": wandb.run.summary.get('test_loss', -1),
                        "test_intensity_accuracy": wandb.run.summary.get('test_intensity_accuracy', -1),
                        "test_location_accuracy": wandb.run.summary.get('test_location_accuracy', -1),
                        "test_quantity_accuracy": wandb.run.summary.get('test_quantity_accuracy', -1),
                        "test_tissue_accuracy": wandb.run.summary.get('test_tissue_accuracy', -1),
                        "test_malignancy_accuracy": wandb.run.summary.get('test_malignancy_accuracy', -1),
                        "lr": scheduler.get_last_lr()[0]
                    })

            # Evaluation step
            if config.batch_size == 16:
                eval_per_step = 33
            elif config.batch_size == 24:
                eval_per_step = 22
            elif config.batch_size == 2:
                eval_per_step = 264
            elif config.batch_size == 1:
                # eval_per_step = 5280
                eval_per_step = 10000
            else:
                eval_per_step = 50  # Default value

            if batch_idx % eval_per_step == 0:
                if rank == 0:
                    # Evaluate on test set after each epoch
                    os.makedirs(os.path.join(config.checkpoint_dir, "validation_output"), exist_ok=True)
                    save_name = os.path.join(config.checkpoint_dir, "validation_output", f"test_results_epoch_{epoch}_step_{batch_idx + start_batch_idx}.csv")
                    test_loss, test_accuracies = evaluate(model, test_dataloader, device, save_name)
                    print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {test_loss:.4f}")
                    for k, v in test_accuracies.items():
                        print(f"Test {k.capitalize()} Accuracy: {v:.4f}")

                    # Log test metrics
                    wandb.log({
                        "epoch": epoch + 1,
                        "batch": batch_idx + start_batch_idx + 1,
                        "num_images_so_far": num_images_so_far,
                        "test_loss": test_loss,
                        **{f"test_{k}_accuracy": v for k, v in test_accuracies.items()}
                    })

                    # Save checkpoint
                    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{batch_idx + start_batch_idx}.pth")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx + start_batch_idx,
                        'shuffle_seed': shuffle_seed,
                        'model_state_dict': model.module.state_dict() if config.distributed else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'test_loss': test_loss,
                        'test_accuracies': test_accuracies,
                        'num_images_so_far': num_images_so_far
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        # Log epoch loss
        if rank == 0:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": epoch_loss / len(dataloader)
            })

            # Save checkpoint at the end of each epoch
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch + 1,
                'batch_idx': -1,  # Indicate end of epoch
                'shuffle_seed': shuffle_seed,
                'model_state_dict': model.module.state_dict() if config.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_loss': test_loss,
                'test_accuracies': test_accuracies,
                'num_images_so_far': num_images_so_far
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"Epoch {epoch+1}/{num_epochs} completed")

    if rank == 0:
        print("Training completed!")
        wandb.finish()


def evaluate(model, dataloader, device, save_name):
    if config.distributed and torch.distributed.get_rank() != 0:
        return
    model.eval()
    total_loss = 0
    correct_predictions = {
        'intensity': 0,
        'location': 0,
        'quantity': 0,
        'tissue': 0,
        'malignancy': 0
    }
    total_samples = 0

    # Lists to store all outputs and ground truth
    all_outputs = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }
    all_ground_truth = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }

    all_predicted_classes = {
        'intensity': [],
        'location': [],
        'quantity': [],
        'tissue': [],
        'malignancy': []
    }

    all_image_urls = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue
            patches, metadata, query_input, caption_output, snomed_text, snomed_code, image_url, staining_intensity, staining_location, staining_quantity, malignancy, tissue_one_hot, cell_type_one_hot = batch
            
            patches = [patch.to(device) for patch in patches]
            staining_intensity = staining_intensity.to(device)
            staining_location = staining_location.to(device)
            staining_quantity = staining_quantity.to(device)
            tissue_one_hot = tissue_one_hot.to(device)
            malignancy = malignancy.to(device)

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                intensity_out, location_out, quantity_out, tissue_out, malignancy_out, A_raw = model(patches, query_input, cell_type_one_hot, phase="test")

                # Compute losses
                intensity_loss = criterion_intensity(intensity_out, torch.argmax(staining_intensity, dim=1))
                location_loss = criterion_location(location_out, torch.argmax(staining_location, dim=1))
                quantity_loss = criterion_quantity(quantity_out, torch.argmax(staining_quantity, dim=1))
                tissue_loss = criterion_tissue(tissue_out, torch.argmax(tissue_one_hot, dim=1))
                malignancy_loss = criterion_malignancy(malignancy_out, torch.argmax(malignancy, dim=1))
                

                total_loss += (intensity_loss + location_loss + quantity_loss + tissue_loss + malignancy_loss).item()

            # Calculate correct predictions
            correct_predictions['intensity'] += (torch.argmax(intensity_out, dim=1) == torch.argmax(staining_intensity, dim=1)).sum().item()
            correct_predictions['location'] += (torch.argmax(location_out, dim=1) == torch.argmax(staining_location, dim=1)).sum().item()
            correct_predictions['quantity'] += (torch.argmax(quantity_out, dim=1) == torch.argmax(staining_quantity, dim=1)).sum().item()
            correct_predictions['tissue'] += (torch.argmax(tissue_out, dim=1) == torch.argmax(tissue_one_hot, dim=1)).sum().item()
            correct_predictions['malignancy'] += (torch.argmax(malignancy_out, dim=1) == torch.argmax(malignancy, dim=1)).sum().item()

            total_samples += staining_intensity.size(0)

            # Store outputs and ground truth
            all_outputs['intensity'].extend(intensity_out.cpu().numpy())
            all_outputs['location'].extend(location_out.cpu().numpy())
            all_outputs['quantity'].extend(quantity_out.cpu().numpy())
            all_outputs['tissue'].extend(tissue_out.cpu().numpy())
            all_outputs['malignancy'].extend(malignancy_out.cpu().numpy())

            all_predicted_classes['intensity'].extend(torch.argmax(intensity_out, dim=1).cpu().numpy())
            all_predicted_classes['location'].extend(torch.argmax(location_out, dim=1).cpu().numpy())
            all_predicted_classes['quantity'].extend(torch.argmax(quantity_out, dim=1).cpu().numpy())
            all_predicted_classes['tissue'].extend(torch.argmax(tissue_out, dim=1).cpu().numpy())
            all_predicted_classes['malignancy'].extend(torch.argmax(malignancy_out, dim=1).cpu().numpy())

            all_ground_truth['intensity'].extend(staining_intensity.cpu().numpy())
            all_ground_truth['location'].extend(staining_location.cpu().numpy())
            all_ground_truth['quantity'].extend(staining_quantity.cpu().numpy())
            all_ground_truth['tissue'].extend(tissue_one_hot.cpu().numpy())
            all_ground_truth['malignancy'].extend(malignancy.cpu().numpy())

            all_image_urls.extend(image_url)

    avg_loss = total_loss / len(dataloader)
    accuracies = {k: v / total_samples for k, v in correct_predictions.items()}

    # Save all outputs and ground truth to a CSV file
    results_df = pd.DataFrame({
        'image_url': all_image_urls,
        'intensity_out': all_outputs['intensity'],
        'intensity_true': all_ground_truth['intensity'],
        'location_out': all_outputs['location'],
        'location_true': all_ground_truth['location'],
        'quantity_out': all_outputs['quantity'],
        'quantity_true': all_ground_truth['quantity'],
        'tissue_out': all_outputs['tissue'],
        'tissue_true': all_ground_truth['tissue'],
        'malignancy_out': all_outputs['malignancy'],
        'malignancy_true': all_ground_truth['malignancy']
    })
    # Save the results
    results_df.to_csv(save_name, index=False)
    print(f"Test results saved to {save_name}")

    # Plot confusion matrices
    def plot_confusion_matrix(y_true, y_pred, title, ax):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        # Get the class names from hpadataset.py
        if 'intensity' in title.lower():
            class_names = ['negative', 'weak', 'moderate', 'strong']
        elif 'location' in title.lower():
            class_names = ['none', 'cytoplasmic/membranous', 'nuclear', 'cytoplasmic/membranous,nuclear']
        elif 'quantity' in title.lower():
            class_names = ['none', '<25%', '25%-75%', '>75%']
        elif 'malignancy' in title.lower():
            class_names = ['normal', 'cancer']
        elif 'tissue' in title.lower():
            class_names = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
                           'bronchus', 'carcinoid', 'caudate', 'cerebellum',
                           'cerebral cortex', 'cervical', 'cervix', 'colon', 'colorectal',
                           'duodenum', 'endometrial', 'endometrium', 'epididymis',
                           'esophagus', 'fallopian tube', 'gallbladder', 'glioma',
                           'head and neck', 'heart muscle', 'hippocampus', 'kidney', 'liver',
                           'lung', 'lymph node', 'lymphoma', 'melanoma', 'nasopharynx',
                           'oral mucosa', 'ovarian', 'ovary', 'pancreas', 'pancreatic',
                           'parathyroid gland', 'placenta', 'prostate', 'rectum', 'renal',
                           'salivary gland', 'seminal vesicle', 'skeletal muscle', 'skin',
                           'small intestine', 'smooth muscle', 'soft', 'spleen', 'stomach',
                           'testis', 'thyroid', 'thyroid gland', 'tonsil', 'urinary bladder',
                           'urothelial', 'vagina']
        else:
            class_names = [str(i) for i in range(cm.shape[0])]

        ax.set_xticks(np.arange(len(class_names)) + 0.5)
        ax.set_yticks(np.arange(len(class_names)) + 0.5)
        ax.set_xticklabels(class_names, rotation=90, ha='right')
        ax.set_yticklabels(class_names, rotation=0)

    # Plot for staining intensity, location, quantity, and malignancy
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    tasks = ['intensity', 'location', 'quantity', 'malignancy']
    for i, task in enumerate(tasks):
        y_true = np.argmax(all_ground_truth[task], axis=1)
        y_pred = np.argmax(all_outputs[task], axis=1)
        plot_confusion_matrix(y_true, y_pred, f'Confusion Matrix - {task.capitalize()}', axes[i//2, i%2])

    plt.tight_layout()
    cm_path = save_name.replace('.csv', '_confusion_matrices.png')
    plt.savefig(cm_path)
    plt.close()

    # Plot for tissue type
    plt.figure(figsize=(30, 30))  # Increased figure size for better readability
    y_true = np.argmax(all_ground_truth['tissue'], axis=1)
    y_pred = np.argmax(all_outputs['tissue'], axis=1)
    plot_confusion_matrix(y_true, y_pred, 'Confusion Matrix - Tissue Type', plt.gca())
    
    tissue_cm_path = save_name.replace('.csv', '_tissue_confusion_matrix.png')
    plt.savefig(tissue_cm_path)
    plt.close()

    return avg_loss, accuracies


def parse_config():
    parser = argparse.ArgumentParser(description='Train HPA-VLM model')
    # parser.add_argument('--config', type=str, default='training_config/config_9.ini', help='Path to the configuration file')
    parser.add_argument('--config', type=str, default='config/config.ini', help='Path to the configuration file')
    # Add arguments for distributed training
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='Local rank for distributed training')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Convert config values to desired data types without defaults
    def convert_value(convert_func, key):
        try:
            return convert_func(config['DEFAULT'][key])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting {key}: {str(e)}. Please check your config file.")

    # Create a dictionary to hold the converted values
    converted_config = {
        'base_model_name': config['DEFAULT']['base_model_name'],
        'shuffle_training': convert_value(lambda x: x.lower() == 'true', 'shuffle_training'),
        'amp': convert_value(lambda x: x.lower() == 'true', 'amp'),
        'freeze_vit': convert_value(lambda x: x.lower() == 'true', 'freeze_vit'),
        'freeze_query_features_encoder': convert_value(lambda x: x.lower() == 'true', 'freeze_query_features_encoder'),
        'batch_size': convert_value(int, 'batch_size'),
        'num_workers': convert_value(int, 'num_workers'),
        'num_epochs': convert_value(int, 'num_epochs'),
        'learning_rate': convert_value(float, 'learning_rate'),
        'weight_decay': convert_value(float, 'weight_decay'),
        'use_cell_type_embedding': convert_value(lambda x: x.lower() == 'true', 'use_cell_type_embedding'),
        'use_cell_type': config['DEFAULT']['use_cell_type'],
        'model_version': config['DEFAULT']['model_version'],
        'checkpoint_dir': config['DEFAULT'].get('checkpoint_dir', 'checkpoints'),
        'load_checkpoint': eval(config['DEFAULT'].get('load_checkpoint')),
        'checkpoint_path': config['DEFAULT'].get('checkpoint_path', ''),
        'distributed': args.distributed,
        'local_rank': args.local_rank,
    }
    
    # Create a run name using base_model_name and learning_rate
    run_name = f"{converted_config['base_model_name']}_lr={converted_config['learning_rate']}"
    if converted_config['freeze_vit']:
        run_name += "_freeze_vit"
    if converted_config['freeze_query_features_encoder']:
        run_name += "_freeze_query_encoder"
    if converted_config['use_cell_type_embedding']:
        run_name += "use_cell_type_embedding"
    if converted_config['load_checkpoint']:
        run_name += "_resumed"
    run_name += f"_rank{converted_config['local_rank']}"

    # Initialize wandb with config and run name
    if not converted_config['distributed'] or converted_config['local_rank'] == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "isight-slide"),
            config=converted_config,
            name=run_name
        )

    return argparse.Namespace(**converted_config)

if __name__ == "__main__":
    config = parse_config()
    set_seed(1001)
    if config.distributed:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(config.local_rank)
    # **Modified code to reflect resumed training in checkpoint_dir**
    if config.load_checkpoint:
        # Use the same run directory as the checkpoint
        run_dir = os.path.dirname(config.checkpoint_path)
        print(f"Resuming from run directory: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
    # Set wandb directory
    os.environ["WANDB_DIR"] = os.path.join(run_dir, "wandb")
    # Update config with new checkpoint directory
    config.checkpoint_dir = run_dir
    print("-------------------------------------------------------------")
    print(config.checkpoint_dir)
    print("-------------------------------------------------------------")

    train(config)
