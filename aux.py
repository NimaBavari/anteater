import hashlib
from pathlib import Path


def generate_directory_checksum(dir_raw: str) -> str:
    """Generates a checksum based on the names and contents of all files within the .originals directory.

    The provided path is guaranteed to be a directory containing at least one supported image file.
    """
    dir_path = Path(dir_raw)
    original_dir = dir_path / ".originals"

    # Use .originals directory if it exists; otherwise, fall back to main directory
    target_dir = original_dir if original_dir.exists() else dir_path

    file_hashes = []
    for item in sorted(target_dir.iterdir()):
        if not item.is_file():
            continue

        if not item.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue

        with open(item, "rb") as f:
            file_content = f.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            file_hashes.append((item.name, file_hash))

    combined_string = "".join(f"{name}{hash_val}" for name, hash_val in file_hashes)
    return hashlib.sha256(combined_string.encode()).hexdigest()
