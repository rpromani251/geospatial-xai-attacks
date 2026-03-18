from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"