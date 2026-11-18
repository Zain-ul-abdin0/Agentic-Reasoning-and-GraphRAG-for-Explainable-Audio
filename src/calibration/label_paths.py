"""Local questionnaire labels are private inputs, not repository assets."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def default_label_files():
    configured = os.environ.get("DAIC_WOZ_LABEL_FILES", "")
    if configured:
        return [Path(value.strip()) for value in configured.split(os.pathsep) if value.strip()]
    private_files = sorted((BASE_DIR / "data/private/labels").glob("*.csv"))
    if private_files:
        return private_files
    names = ("train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv", "full_test_split.csv")
    folders = (BASE_DIR / "documentations", BASE_DIR.parent / "Documentations")
    return [folder / name for folder in folders for name in names if (folder / name).is_file()]
