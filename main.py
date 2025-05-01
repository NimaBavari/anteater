import logging
import os
import sys

import napari

from annotator import ImageAnnotator


def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with a console handler."""
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


def main() -> None:
    """Bootstraps the application."""
    logger = setup_logger(__name__)
    if len(sys.argv) != 2:
        logger.info("Usage: python main.py <image_directory>.")
        sys.exit(1)

    image_dir = sys.argv[1]
    if not os.path.isdir(image_dir):
        logger.error(f"{image_dir} is not a valid directory.")
        sys.exit(1)

    try:
        _ = ImageAnnotator(image_dir, logger)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

    napari.run()


if __name__ == "__main__":
    main()
