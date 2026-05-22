# APTOS 2019 Diabetic Retinopathy — ConvNeXt

Group project for the APTOS 2019 Blindness Detection task. Goal: beat the 2019
Kaggle leaderboard winners using a modern (ConvNeXt) backbone, a hybrid
Cross-Entropy + Soft-QWK loss, and a training recipe informed by the
1st/4th place writeups.

## Layout

```
baseline/   ResNet50 + MSE regression baseline (reference)
convnext/   ConvNeXt-Tiny + HybridCEKappaLoss (main model)
```

## Setup

1. Install Python 3.10+ and a CUDA-capable PyTorch build.
2. Install dependencies:

   ```
   pip install torch torchvision pandas scikit-learn pillow tqdm albumentations
   ```

3. Download the APTOS 2019 dataset from Kaggle:
   <https://www.kaggle.com/competitions/aptos2019-blindness-detection/data>

   Place it so the project looks like:

   ```
   <project root>/
     train.csv
     test.csv
     processed_train_images/<id_code>.png
     processed_test_images/<id_code>.png
   ```

   (We pre-resize images offline. The model also re-resizes to 512×512 at load
   time, so any reasonably square crop works.)

4. **Update the hard-coded paths** at the top of each script to match your
   machine:

   - [convnext/convNext.py](convnext/convNext.py): `CSV_PATH`, `IMAGE_DIR`,
     `CHECKPOINT_PATH`
   - [baseline/train_baseline.py](baseline/train_baseline.py): `TRAIN_CSV`,
     `IMAGE_DIR`, `CHECKPOINT_DIR`

## Running

Baseline (ResNet50 + MSE regression):

```
python baseline/train_baseline.py
```

Main model (ConvNeXt-Tiny + HybridCEKappa, 30 epochs):

```
python convnext/convNext.py
```

The ConvNeXt script saves the best checkpoint (selected by **validation
quadratic-weighted kappa**) to `convnext/convnext_best.pth`.

## Training recipe notes

- **Loss**: `HybridCEKappaLoss` = α·CE + (1−α)·SoftQWK, with α scheduled by
  `get_alpha(epoch)` — pure CE warmup for 5 epochs, then linear decay to
  α=0.2 over the remaining 25 epochs.
- **Augmentations**: Albumentations (shift/scale/rotate, brightness/contrast,
  HSV, coarse dropout). The 4th-place 2019 team relied heavily on this set.
- **No Ben Graham preprocessing**: both 1st and 4th place 2019 teams reported
  it did not help; ConvNeXt is strong enough to learn through lighting
  variation directly.
- **Class weights**: APTOS is ~50% class-0; weights are computed via
  `sklearn.utils.class_weight.compute_class_weight("balanced")`.

## Open extensions (not yet implemented)

These are the two biggest remaining score gaps per the 1st/4th place writeups:

1. **Stage-A: 2015 DR pretrain** — pretrain on the older, much larger 2015
   Diabetic Retinopathy dataset before fine-tuning on APTOS 2019. Insertion
   point marked in `convnext/convNext.py`.
2. **Stage-C: pseudo-labeling** — after Stage-B converges, predict on the
   unlabeled test set, keep high-confidence rows, retrain. Stub function
   `pseudo_label()` in `convnext/convNext.py`.
