"""Approach A1: frozen DINOv3 linear probe for Severstal defect segmentation.

Backbone stays frozen (default for this model); only the linear decoder head
trains. Cheap, fast, tests whether DINOv3's pretraining already transfers to
this domain with zero backbone adaptation.
"""
import lightly_train

BASE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints\a1_frozen"

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out=OUT_DIR,
        model="dinov3/vits16-linear",
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
        batch_size=16,
        num_workers=0,
        model_args={"backbone_freeze": True},
        overwrite=True,
    )
