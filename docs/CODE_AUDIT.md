# Code audit before release

Every defect found while preparing this repository, what it affected, and what was changed.
Written so a reviewer can check each claim rather than take it on trust.

**No published number changes.** Every fix below is either bit-identical under the released
configuration, or affects a code path the published runs never took. Where a fix would change
behaviour under some *other* configuration, that is stated explicitly.

---

## iSight-slide

### 1. Cell-type embedding indexed with a leaked loop variable — REAL BUG

`model/patch_encoder_with_clam.py`

```python
for ix, patch_tensor in enumerate(patches):     # line ~209
    ...
                                                 # loop ends, ix == len(patches) - 1
cell_type_embed = cell_type_embed[ix].unsqueeze(0).expand(M.size(0), -1)   # line ~276
```

`ix` is the loop variable left over from patch encoding. After the loop it equals
`batch_size - 1`, so **every sample in the batch received the last sample's cell-type
embedding**, broadcast by `.expand`.

**Not triggered by the published model.** `config/config.ini` ships `batch_size = 1`, where
`ix == 0` is that one sample's own row. The paper's model version is `v3_all_tokens`; the other
configs carrying it in the original tree ran at batch sizes 16 and 24, where this bug is live.

**Fixed** by indexing per sample and asserting the batch dimensions agree. Bit-identical at
`batch_size = 1`; changes behaviour only at `batch_size > 1`, where the original was wrong.
Left unfixed, anyone raising the batch size would have silently mis-conditioned the model with
no error raised.

### 2. Text dropout applied per batch, not per sample

Same file. The original drew one Bernoulli for the whole batch:

```python
if random.random() < 0.5:      # one draw for the entire batch
    M_single_token = M_single_token + query_features
```

so at `batch_size > 1` the text branch was on for everyone or off for everyone. At
`batch_size = 1` — the shipped config — per-batch and per-sample are the same thing, so the
published model is unaffected.

**Fixed** to a per-sample mask, with the keep probability as the named constant
`TEXT_KEEP_PROB = 0.5`. `force_query` still forces the branch on for every sample.

### 3. `datadir` used but never defined — REAL BUG

`main_v2_dist_resume.py`. The line defining `datadir` was commented out, but the
`simple_downsample` branch reads it, so selecting that model version raised `NameError`
before the first step. The MIL versions the paper uses never touch it, which is why the
published runs were unaffected — and why the bug survived.

**Fixed**: defined from `ISIGHT_IMAGE_DIR`.

### 4. Learning-rate scheduler stepped on an unreduced per-rank loss under DDP

`main_v2_dist_resume.py`

```python
scheduler.step(total_loss.item())    # once per BATCH, this rank's loss only
```

Two issues. `ReduceLROnPlateau` is designed to be stepped once per epoch on a held-out metric;
here it is stepped every batch (with `patience=500`, i.e. 500 batches). And `total_loss.item()`
is the local rank's loss, so under DDP different ranks can cut the learning rate at different
steps and the optimisers drift apart.

**Fixed**: the loss is all-reduced so every rank steps on the same number. The *cadence* is
deliberately unchanged (still per batch) so the published run reproduces exactly; set
`SCHEDULER_PER_EPOCH=1` for the conventional schedule.

### 5. Dead code that misrepresents the aggregation

```python
M_single_token = M[:, 0, :]            # CLS token      <- computed, then discarded
M_single_token = torch.mean(M, dim=1)  # mean of all tokens
```

The CLS line never affected anything; the model has always used the mean over all tokens,
which is the "all-token" aggregation the paper describes. **Removed**, so the code cannot be
misread as CLS pooling.

### 6. Hard-coded paths to one cluster, including another lab's directories

`/project/kimlab_tcga/jleiby/IHC/data/...` (training and validation metadata, RLE masks),
`/project/zhihuanglab/zhi/HPA-VLM/...`, and `/project/XXXXX` placeholders in `config.ini`.
Nothing would run outside the original machine — plausibly the reason the code read as
incomplete.

**Fixed**: `ISIGHT_DATA_ROOT`, `ISIGHT_TRAIN_META`, `ISIGHT_TEST_META`, `ISIGHT_RLE_DIR`,
`ISIGHT_RLE_INDEX`, `ISIGHT_IMAGE_DIR`.

---

## iSight-cell

One defect that made the code unrunnable outside the original cluster (§9), one
specification the code did not implement (§8), and one thing a reader should know (§7):

### 7. `loc_none` is a residual, and it absorbs head disagreement

`clinical_features/upenn_melanoma/build_cell_melan.py`

```python
stl = loc[inten > 0]                                  # positive cells only -- correct
f[8], f[9], f[10] = nuc, cyto, mixed fractions
f[11] = max(0, 1 - f[8] - f[9] - f[10])               # loc_none, by subtraction
```

Conditioned on "positive", a `none` localisation should be empty — a stained cell has a
staining shape. It is non-zero for all 194 patients (median 0.19% of positive cells, max 2.96%)
because the intensity and location heads are independent softmaxes with nothing tying them
together. The residual silently absorbs exactly that disagreement, and `max(0, ...)` would also
hide any negative value.

It tracks negativity rather than biology: `corr(loc_none, g_neg) = +0.755`, R² = 0.571 against
`g_neg` alone. `marker_status.csv` independently flags it `REDUNDANT(=t_neg)`.

**Not silently changed here.** The feature builder is released as it ran. The figure scripts
drop `loc_none` and renormalise the three real classes over `{nuclear, cytoplasmic, cyto+nuc}`,
which is exact because all four shared one denominator, and the effect is small
(corr(new, old) ≥ 0.9996, max change 2.3 points).

### 8. Early stopping was specified but not implemented — now implemented

The selection rule for the staining model was written down before release:

```
criterion   validation macro-F1  (mean of per-cell intensity and location F1)
min_delta   0.001  (0.1%)
patience    2
semantics   keep-current
```

It was **not in the training code**: `grep -cE "patience|min_delta|early_stop"` over the
trainers returned 0, both rounds ran their full epoch budget (15/15 and 20/20), every epoch was
checkpointed, and the rule was applied afterwards to `history.csv` to pick epoch 9.

**Fixed.** The rule is now implemented in `isight_cell/code/pipeline/earlystop.py` and wired
into both trainers (`train_foundation.py` for step 1, `train_shards_resample.py` for step 2).
It runs by default; `EARLY_STOP=0` restores the full-budget behaviour, and `ES_METRIC`,
`ES_MIN_DELTA`, `ES_PATIENCE` are overridable. Under DDP the decision is made on rank 0 and
broadcast, so all ranks leave the loop together, and the counter is replayed from `history.csv`
when a run resumes. On stopping, the trainer writes `selected_model.pt` — the **current** epoch,
not the best-so-far — and the final test evaluation loads it in preference to `best_model.pt`.

**It reproduces the released checkpoint.** Replaying the published run's
`runs/round3_resample_s1_lr2e5/history.csv` through this implementation stops at **epoch 9**:

```
ep07  0.813259  improve -> best, threshold 0.814259
ep08  0.814246  misses the threshold by 1.3e-05  -> bad=1
ep09  0.813690  declines                         -> bad=2 -> stop, keep ep09
```

Two caveats a reader should carry:

* **The margin at ep08 is 1.3e-05.** A different seed, GPU count or reduction order can flip it.
  The same rule on the other logged validation metrics stops at ep09 for `val_cell_loc_f1` and
  `val_img_loc_f1`, at ep12 for `val_cell_int_f1` and at ep07 for `val_img_int_f1` — so three of
  the metrics agree on ep09, and the rule is not equally stable across all of them.
* **argmax over all 20 epochs would give ep16**, which is no worse than ep09 on any of the six
  validation metrics (largest gap: img loc-F1 +0.0080). ep09 is what the early-stopping rule
  returns, not the global validation optimum, and the methods text should say early stopping
  rather than best-checkpoint selection.

### 9. The cell code could not run outside the original cluster — REAL BUG

Every script under `isight_cell/` began by hard-coding one absolute project root and then
inserting three sibling repositories onto `sys.path` to reach the model definition:

```python
ISIGHT = Path("/vast/projects/.../iSight")
for p in ("hnc_immune_markers/code", "iSight_prostate_finetune/code", "iSight_train_model/code"):
    sys.path.insert(0, str(ISIGHT / p))
import train_finetune as T                       # <- the model class, in another repository
from backbones.vision_encoders.uni2 import build_uni2
```

None of those three repositories was part of the release, so **`Net`, `crops_to_input` and the
UNI2-h builder — the model itself — were missing**, and every script raised `ModuleNotFoundError`
on import. This is the largest part of what read as incomplete code.

**Fixed.** The model definition and the UNI2-h backbone are vendored into
`isight_cell/code/deps/`, and every script now resolves them relative to its own location. The
project root is the `ISIGHT_ROOT` environment variable, and the UNI2-h weights come from
`UNI2_CKPT_PATH` (previously a hard-coded path to a local copy) which now raises a clear error
if unset instead of failing inside `torch.load`.

`deps/train_finetune.py` additionally imported four constants (ImageNet normalisation, input
size, embedding width) from a shared module that also loads the DINOv2 code. They are inlined,
so the training path no longer depends on DINOv2 at all.

### 10. Target-cell classes were resolved through a chain of index spaces

The deployed target-cell classifier has 43 heads, but the class index reached them through two
hops: the per-image `head_idx` stored in the data, remapped by a table carried inside the
checkpoint, into a head column. The trainer derived that table at run time from whichever
classes survived a `--drop_classes` flag, so the head order was a property of one particular
invocation rather than anything written down, and the selection script defaulted to a
*different* checkpoint with a different head count than the one the paper used.

**Fixed.** The 43 classes are now a shipped table, `isight_cell/meta/classes_43.csv`
(`cls43, head_idx, tissue, cell_type`), and it is applied once at crop extraction: `class_idx`
is written directly in the 0..42 space, the trainer builds exactly 43 heads with no remap, and
the selector reads the same table instead of deriving anything. `CKPT` is now required rather
than defaulting to the wrong checkpoint.

The table was verified against the deployed checkpoint before being baked in: its 43 entries are
identical to the `old_to_new` mapping stored in `target_cell_43cls/best.pt`, the head columns are
contiguous 0..42, and `head.W` has shape `(43, 1536, 2)`. The selector still asserts this
agreement whenever a checkpoint carries the table inline, so a mismatched pair fails loudly
rather than silently scoring cells with the wrong head.

---

## What was checked and found clean

* Both files parse, and a static pass over `main_v2_dist_resume.py` resolves every load-time
  name (the one remaining hit is a lambda parameter).
* Training loop mechanics: `optimizer.zero_grad()` present, AMP `scale/step/update` ordering
  correct, `train_sampler.set_epoch(epoch)` called, checkpoint saves `model.module` under DDP.
* Losses: raw logits into `CrossEntropyLoss`, no double softmax.
* Reader-study cell selection: all 151,508 kept cells across the 66 images lie inside the
  tissue mask — verified at the centroid, and on a 5-image sample at every contour pixel.

## Not shipped

`code/target_cell/dinov2_code/` — a vendored copy of Meta's DINOv2. Third-party code, released
under its own licence; the repository should depend on it rather than redistribute it.
