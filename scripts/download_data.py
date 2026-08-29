from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
TARGET_NAME = "MetroPT3(AirCompressor).csv"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / TARGET_NAME


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"Dataset already exists: {TARGET}")
        return

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive = Path(tmp.name)
    try:
        print("Downloading MetroPT-3 from UCI...")
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        with zipfile.ZipFile(archive) as zf:
            matches = [name for name in zf.namelist() if name.endswith(TARGET_NAME)]
            if not matches:
                raise RuntimeError(f"{TARGET_NAME} not found in UCI archive")
            with zf.open(matches[0]) as source, TARGET.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        print(f"Saved dataset to {TARGET}")
    finally:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
