"""Approach A1b: frozen DINOv3 backbone + EoMT decoder (Mask2Former-style,
mask+dice+class Hungarian-matching loss) for Severstal defect segmentation.

Compared to A1 (bare linear probe): much stronger decoder, and a loss with a
real Dice component - both better suited to the class imbalance that made
A1 fail completely on the rare defect classes (1, 2).
"""
import lightly_train

BASE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints\a1b_eomt_frozen"

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-eomt",
        data={
            "train": {
                "images": f"{BASE_DIR}/images/train",
                "masks": f"{BASE_DIR}/masks_split/train",
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
        steps=4000,
        batch_size=2,
        num_workers=0,
        model_args={"backbone_freeze": True},
        logger_args={"val_every_num_steps": 2000},
        overwrite=True,
    )
