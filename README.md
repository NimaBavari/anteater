# Anteater

_by Tural Mahmudov <nima.bavari@gmail.com>_

Image annotator GUI.

## Overview
- Clean, typechecked, documented codebase
- Clean design
- Full unit test coverage
- `stdout` logging and helpful GUI messages
- **All the required features, as well as the bonus feature, are implemented**
- Minimal environment: no poetry/conda, no containers; just a simple venv
- No deployment strategies

## Usage

All the following commands are run from the project root.

### Set Up Environment
Run:

```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

to set up your environment.

Optionally, you can also set your pre-commit hook by running:

```console
make set-pre-commit
```

### Start Up
Once your environment is ready and activated, you can start the application by running:

```console
python main.py <image_directory>
```

where `<image_directory>` is relative to the project root. Note that it must be a valid directory and must contain at least one supported image file. For example,

```console
python main.py ../heidelberg-april-2025/
```

### Beautify
Run:

```console
make code-quality
```

to lint, format, and typecheck the source code.

### Testing
In order to run the tests, run from within your venv:

```console
python -m unittest discover -v
```

## Storage Format and Strategy

The annotation storage system uses JSON files as the primary format for storing image annotations. Each image has a corresponding JSON file stored in a dedicated annotation directory, ensuring a robust and scalable approach to managing annotations.

### Storage Format
- **Format**: JSON
- **Structure**: Each JSON file is named after the image's stem (filename without extension) and contains two key fields:
  - `labels`: A list of strings representing the hierarchical annotation (e.g., `["Animal", "Dog", "Labrador"]`).
  - `is_transformed`: A boolean indicating whether a transformation (e.g., Gaussian blur) has been applied to the image.
- **File Naming**: For an image named `example.png`, the annotation file is `example.json`.

### Storage Strategy
- **Directory Structure**: Annotations are stored in a user-specific directory under `~/.annotations/<dir_id>`, where `<dir_id>` is a unique checksum generated from the image directory path. This ensures isolation of annotations for different image sets.
- **One File per Image**: Each image has a dedicated JSON file, which simplifies access, reduces contention, and supports incremental updates without affecting other annotations.
- **Backup Mechanism**: Original images are backed up in a `.originals` subdirectory within the image directory to preserve unmodified data, ensuring that transformations (e.g., Gaussian blur) can be reverted.
- **Robustness**: The use of JSON ensures human-readable, platform-independent storage. Error handling in the `AnnotationManager` class catches and logs issues during loading or saving, preventing data corruption and providing feedback via the UI.

### Rationale
- **JSON Choice**: JSON is lightweight, widely supported, and suitable for structured data like hierarchical labels and transformation states. It integrates seamlessly with Python's `json` module.
- **Per-Image Files**: Individual files per image enhance modularity, making it easier to manage large datasets and support parallel processing or distributed systems in the future.
- **Directory Isolation**: Using a checksum-based directory prevents conflicts when annotating images from different directories with similar filenames.

This approach balances simplicity, robustness, and extensibility, making it suitable for both small and large-scale image annotation tasks.

