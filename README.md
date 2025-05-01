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
