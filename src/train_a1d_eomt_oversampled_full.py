"""Approach A1d: same as A1c (frozen DINOv3 + EoMT decoder + oversampled
train set), but run to Lightly's own documented EoMT convergence benchmark
(~31 epochs, derived from their ADE20K default: 20210 images / batch 16 * 31
epochs ~= 40k steps) instead of the earlier 4000-step quick check.

31 epochs on our 12,678-image oversampled train set:
  31 * 12678 / 16 = 24564 steps
"""
import lightly_train

BASE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints\a1d_eomt_oversampled_full"

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-eomt",
        data={
            "train": {
                "images": f"{BASE_DIR}/images/train_oversampled",
                "masks": f"{BASE_DIR}/masks_split/train_oversampled",
            },
            "val": {
                "images": f"{BASE_DIR}/images/val",
                "masks": f"{BASE_DIR}/masks_split/val",
            },
            "classes": {
                0: "background",
                1: "defect_1",
                2: "defect_2",
                3: "defect_3",
                4: "defect_4",
            },
        },
        steps=24564,
        batch_size=2,
        num_workers=0,
        model_args={"backbone_freeze": True},
        logger_args={"val_every_num_steps": 6000},
        overwrite=True,
    )
