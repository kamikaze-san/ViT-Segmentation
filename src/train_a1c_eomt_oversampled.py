"""Approach A1c: same as A1b (frozen DINOv3 + EoMT decoder), but trained on
the hardlink-oversampled train set to test whether directly increasing rare
class exposure fixes the class 1/2 collapse seen in A1a and A1b. Val/test
splits are untouched (only train exposure changes).

Same step budget as A1b (4000 steps) to isolate the effect of oversampling.
"""
import lightly_train

BASE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints\a1c_eomt_oversampled"

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
        steps=4000,
        batch_size=2,
        num_workers=0,
        model_args={"backbone_freeze": True},
        logger_args={"val_every_num_steps": 2000},
        overwrite=True,
    )
