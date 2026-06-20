from pathlib import Path


TOP_K_FOR_PURITY = 10


BASE_FEATURE_DIR = Path(
    "/home/woody/rlvl/rlvl172v/revelio/SD-kSAE/features"
)


DATASET_LAYER_DIRS = {
    "Caltech101": {
        "mid_block": (
            BASE_FEATURE_DIR
            / "caltech101"
            / "SDv1-5"
            / "step25_mid"
        ),
        "up_ft0": (
            BASE_FEATURE_DIR
            / "caltech101"
            / "SDv1-5"
            / "step25_upft0"
        ),
        "up_ft1": (
            BASE_FEATURE_DIR
            / "caltech101"
            / "SDv1-5"
            / "step25_upft1"
        ),
        "up_ft2": (
            BASE_FEATURE_DIR
            / "caltech101"
            / "SDv1-5"
            / "step25_upft2"
        ),
    },

    "Oxford-IIIT Pet": {
        "mid_block": (
            BASE_FEATURE_DIR
            / "oxfordpet"
            / "SDv1-5"
            / "step25_mid"
        ),
        "up_ft0": (
            BASE_FEATURE_DIR
            / "oxfordpet"
            / "SDv1-5"
            / "step25_upft0"
        ),
        "up_ft1": (
            BASE_FEATURE_DIR
            / "oxfordpet"
            / "SDv1-5"
            / "step25_upft1"
        ),
        "up_ft2": (
            BASE_FEATURE_DIR
            / "oxfordpet"
            / "SDv1-5"
            / "step25_upft2"
        ),
    },
}
