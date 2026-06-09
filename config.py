from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("THESIS_DATA_DIR", BASE_DIR / "data")).expanduser()

HDMSPECTRA_H5 = Path(
    os.getenv("HDMSPECTRA_H5", DATA_DIR / "HDMSpectra.hdf5")
).expanduser()

FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Put the file there locally, or set the corresponding path in config.py."
        )
    return path