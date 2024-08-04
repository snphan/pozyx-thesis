from pathlib import Path
import shutil

if __name__ == "__main__":
    ROOT_DIR = Path()
    for path in ROOT_DIR.glob("*.csv"):
        FOLDER_DIR = ROOT_DIR.joinpath(path.name.split("_")[0])
        if not FOLDER_DIR.exists():
            FOLDER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, FOLDER_DIR)
        print(f"copied {path} to {FOLDER_DIR}")
    