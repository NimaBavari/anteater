from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as imageio
import napari
import numpy as np
from magicgui.widgets import ComboBox, Container
from qtpy.QtWidgets import QCheckBox, QLabel, QListWidget, QPushButton, QTextEdit, QVBoxLayout, QWidget
from scipy.ndimage import gaussian_filter

from aux import generate_directory_checksum
from constants import CLASS_HIERARCHY


class ImageManager:
    """Unit for loading and retrieving image files from a directory."""

    def __init__(self, image_dir: str, logger: logging.Logger) -> None:
        self.image_dir = Path(image_dir)
        self.original_dir = self.image_dir / ".originals"
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.image_files: List[Path] = [
            f for f in self.image_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ]
        if not self.image_files:
            raise Exception(f"No supported image files found in '{image_dir}'.")
        self._backup_originals()

    def _backup_originals(self) -> None:
        """Creates a backup of original images in the .originals directory."""
        for image_file in self.image_files:
            original_path = self.original_dir / image_file.name
            if not original_path.exists():
                shutil.copy2(image_file, original_path)
                self.logger.info(f"Backed up original image to '{original_path}'.")

    def get_image(self, index: int) -> Optional[np.ndarray]:
        """Retrieves the image data for the given index."""
        if 0 <= index < len(self.image_files):
            try:
                img = imageio.imread(self.image_files[index])
                self.logger.info(f"Loaded image '{self.image_files[index]}' with shape {img.shape}, dtype {img.dtype}.")
                return np.asarray(img, dtype=np.uint8)
            except Exception as e:
                self.logger.error(f"Failed to load image '{self.image_files[index]}': {e}")
                return None
        return None

    def get_original_image(self, index: int) -> Optional[np.ndarray]:
        """Retrieves the original image data for the given index."""
        if 0 <= index < len(self.image_files):
            original_path = self.original_dir / self.image_files[index].name
            try:
                img = imageio.imread(original_path)
                self.logger.info(f"Loaded original image '{original_path}' with shape {img.shape}, dtype {img.dtype}.")
                return np.asarray(img, dtype=np.uint8)
            except Exception as e:
                self.logger.error(f"Failed to load original image '{original_path}': {e}")
                return None
        return None

    def get_image_name(self, index: int) -> Optional[str]:
        """Retrieves the name of the image file for the given index."""
        if 0 <= index < len(self.image_files):
            return self.image_files[index].name
        return None

    def save_image(self, index: int, image_data: np.ndarray) -> None:
        """Saves the image data to the original file."""
        if 0 <= index < len(self.image_files):
            try:
                image_path = self.image_files[index]
                imageio.imwrite(image_path, image_data.astype(np.uint8))
                self.logger.info(
                    f"Saved image to '{image_path}' with shape {image_data.shape}, dtype {image_data.dtype}."
                )
            except Exception as e:
                self.logger.error(f"Failed to save image to '{image_path}': {e}")


class AnnotationManager:
    """Unit for loading, saving, and retrieving annotations."""

    def __init__(self, annotation_dir: Path, annotator: ImageAnnotator) -> None:
        self.annotation_dir = annotation_dir
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.annotator = annotator
        self.annotator.logger.info(f"Annotation directory set to '{self.annotation_dir}'.")
        self.annotations: Dict[str, Tuple[List[str], bool]] = {}

    def load_annotations(self, image_files: List[Path]) -> None:
        """Loads existing annotations from JSON files for the given image files."""
        for image_file in image_files:
            annotation_path = self.annotation_dir / f"{image_file.stem}.json"
            if annotation_path.exists():
                try:
                    with open(annotation_path, "r") as f:
                        data = json.load(f)
                        annotation = data.get("labels", [])
                        is_transformed = data.get("is_transformed", False)
                        self.annotations[image_file.name] = (annotation, is_transformed)
                    self.annotator.logger.info(f"Loaded annotation for '{image_file.name}'.")
                    self.annotator.ui_msgbar.setText(f"Loaded annotation for '{image_file.name}'.")
                except Exception as e:
                    self.annotator.logger.error(f"Failed to load annotation '{annotation_path}': {e}")
                    self.annotator.ui_msgbar.setText(f"Failed to load annotation '{annotation_path}'.")

    def save_annotation(self, image_name: str, annotation: List[str], is_transformed: bool) -> None:
        """Saves the annotation and transformation state for the given image name."""
        annotation_path = self.annotation_dir / f"{Path(image_name).stem}.json"
        try:
            with open(annotation_path, "w") as f:
                json.dump({"labels": annotation, "is_transformed": is_transformed}, f)
            self.annotator.logger.info(
                f"Saved annotation for '{image_name}': {annotation}, is_transformed: {is_transformed}."
            )
            self.annotator.ui_msgbar.setText(f"Saved annotation for '{image_name}': {annotation}.")
        except Exception as e:
            self.annotator.logger.error(f"Failed to save annotation for '{image_name}': {e}")
            self.annotator.ui_msgbar.setText(f"Failed to save annotation for '{image_name}'.")

    def get_annotation(self, image_name: str) -> Tuple[List[str], bool]:
        """Retrieves the annotation and transformation state for the given image name."""
        return self.annotations.get(image_name, ([], False))


class UIComponent:
    """Unit for the user interface components for annotation."""

    def __init__(self, annotator: ImageAnnotator) -> None:
        self.annotator = annotator
        self.image_names = [f.name for f in self.annotator.image_manager.image_files]

        self.category_combo = ComboBox(choices=["Animal", "Other"], label="Category")
        self.category_combo.changed.connect(self.on_category_changed)

        self.subcategory_combo = ComboBox(choices=[], label="Subcategory", enabled=False)
        self.subcategory_combo.changed.connect(self.on_subcategory_changed)

        self.breed_combo = ComboBox(choices=[], label="Breed", enabled=False)

        self.image_list = QListWidget()

        self.transform_checkbox = QCheckBox("Apply Gaussian Blur")

        self.annotator.ui_msgbar.setReadOnly(True)

        self.setup_ui()

        self.annotator.logger.info("UI components initialized.")
        self.annotator.ui_msgbar.setText("UI components initialized.")

    def setup_ui(self) -> None:
        """Sets up the UI components in the napari viewer."""
        dock_widget = QWidget()
        layout = QVBoxLayout()
        dock_widget.setLayout(layout)

        self.image_list.addItems(self.image_names)
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        layout.addWidget(QLabel("Images:"))
        layout.addWidget(self.image_list)

        annotation_container = Container(widgets=[self.category_combo, self.subcategory_combo, self.breed_combo])
        layout.addWidget(annotation_container.native)

        layout.addWidget(self.transform_checkbox)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.annotator.save_current_annotation)
        layout.addWidget(save_button)

        self.annotator.viewer.window.add_dock_widget(dock_widget, area="right", name="Annotation")

        self.annotator.viewer.window.add_dock_widget(self.annotator.ui_msgbar, area="bottom", name="Log")

    def on_image_selected(self, index: int) -> None:
        """Handles image selection from the list widget."""
        self.annotator.on_image_selected(index)

    def on_category_changed(self, value: str) -> None:
        """Updates subcategory options based on category selection."""
        self.subcategory_combo.choices = []
        self.subcategory_combo.enabled = False
        self.breed_combo.choices = []
        self.breed_combo.enabled = False
        self.subcategory_combo.native.setCurrentIndex(-1)
        self.breed_combo.native.setCurrentIndex(-1)
        if value == "Animal":
            self.subcategory_combo.choices = list(CLASS_HIERARCHY["Animal"].keys())
            self.subcategory_combo.enabled = True

    def on_subcategory_changed(self, value: str) -> None:
        """Updates breed options based on subcategory selection."""
        self.breed_combo.choices = []
        self.breed_combo.enabled = False
        self.breed_combo.native.setCurrentIndex(-1)
        if value in ("Cat", "Dog"):
            self.breed_combo.choices = CLASS_HIERARCHY["Animal"][value]
            self.breed_combo.enabled = True

    def update_ui(self, annotation: List[str], is_transformed: bool) -> None:
        """Updates the UI based on the given annotation and transformation state."""
        self.category_combo.value = "Other"
        self.subcategory_combo.choices = []
        self.subcategory_combo.enabled = False
        self.breed_combo.choices = []
        self.breed_combo.enabled = False
        self.subcategory_combo.native.setCurrentIndex(-1)
        self.breed_combo.native.setCurrentIndex(-1)

        if annotation:
            category = annotation[0]
            if category in ["Animal", "Other"]:
                self.category_combo.value = category
                self.on_category_changed(category)
                if len(annotation) > 1 and category == "Animal":
                    subcategory = annotation[1]
                    if subcategory in CLASS_HIERARCHY["Animal"]:
                        self.subcategory_combo.value = subcategory
                        self.on_subcategory_changed(subcategory)
                        if len(annotation) > 2 and subcategory in ("Cat", "Dog"):
                            breed = annotation[2]
                            if breed in CLASS_HIERARCHY["Animal"][subcategory]:
                                self.breed_combo.value = breed

        self.transform_checkbox.setChecked(is_transformed)
        self.annotator.logger.info(f"UI updated with annotation: {annotation}, is_transformed: {is_transformed}.")
        self.annotator.ui_msgbar.setText(f"UI updated with annotation: {annotation}.")

    def get_current_annotation(self) -> List[str]:
        """Retrieves the current annotation from the UI."""
        annotation = [self.category_combo.value]
        if self.category_combo.value == "Animal" and self.subcategory_combo.value:
            annotation.append(self.subcategory_combo.value)
            if self.subcategory_combo.value in ("Cat", "Dog") and self.breed_combo.value:
                annotation.append(self.breed_combo.value)
        return annotation

    def set_current_image_index(self, index: int) -> None:
        """Sets the current image index in the image list."""
        self.image_list.setCurrentRow(index)


class ImageAnnotator:
    """Image annotation coordinator."""

    def __init__(self, image_dir: str, logger: logging.Logger) -> None:
        self.logger = logger
        try:
            self.image_manager = ImageManager(image_dir, self.logger)
        except Exception:
            raise

        dir_id = generate_directory_checksum(image_dir)
        self.annotation_manager = AnnotationManager(Path.home() / f".annotations/{dir_id}", self)
        self.viewer = napari.Viewer()
        self.viewer.title = "Auto1 Image Annotator"
        self.current_image_index = -1
        self.ui_msgbar = QTextEdit()
        self.ui_component = UIComponent(self)
        self.annotation_manager.load_annotations(self.image_manager.image_files)
        if self.image_manager.image_files:
            self.current_image_index = 0
            self.update_image_and_ui()
        else:
            self.logger.info("No images available to annotate.")
            self.ui_msgbar.setText("No images available to annotate.")

        @self.viewer.bind_key("Left")
        def prev_image(_: Any) -> None:
            """Navigate to the previous image."""
            if self.current_image_index > 0:
                self.save_current_annotation()
                self.current_image_index -= 1
                self.update_image_and_ui()

        @self.viewer.bind_key("Right")
        def next_image(_: Any) -> None:
            """Navigate to the next image."""
            if self.current_image_index < len(self.image_manager.image_files) - 1:
                self.save_current_annotation()
                self.current_image_index += 1
                self.update_image_and_ui()

    def update_image_and_ui(self) -> None:
        """Updates the displayed image and UI elements."""
        image = self.image_manager.get_image(self.current_image_index)
        image_name = self.image_manager.get_image_name(self.current_image_index)
        if image is not None:
            annotation, is_transformed = self.annotation_manager.get_annotation(image_name)
            if is_transformed:
                if image.ndim == 3 and image.shape[-1] in (3, 4):
                    blurred = np.zeros_like(image)
                    for channel in range(image.shape[-1]):
                        blurred[..., channel] = gaussian_filter(image[..., channel], sigma=5)
                    image = blurred
                else:
                    image = gaussian_filter(image, sigma=5)
                image = image.astype(np.uint8)
            self.logger.info(f"Displaying image '{image_name}' with shape {image.shape}, dtype {image.dtype}.")
            if self.viewer.layers:
                self.viewer.layers[0].data = image
            else:
                self.viewer.add_image(image, name=image_name)
            self.ui_component.update_ui(annotation, is_transformed)
            self.ui_component.set_current_image_index(self.current_image_index)
            self.logger.info(f"Updated image and UI for '{image_name}'.")
            self.ui_msgbar.setText(f"Updated image and UI for '{image_name}'.")
        else:
            self.logger.error(f"Could not update image at index {self.current_image_index}.")
            self.ui_msgbar.setText(f"Could not update image at index {self.current_image_index}.")

    def save_current_annotation(self) -> None:
        """Saves the current annotation and applies transformation if checked."""
        if 0 <= self.current_image_index < len(self.image_manager.image_files):
            image_name = self.image_manager.get_image_name(self.current_image_index)
            annotation = self.ui_component.get_current_annotation()
            is_transformed = self.ui_component.transform_checkbox.isChecked()
            self.annotation_manager.save_annotation(image_name, annotation, is_transformed)
            if self.viewer.layers:
                if is_transformed:
                    original_data = self.image_manager.get_original_image(self.current_image_index)
                    if original_data is not None:
                        if original_data.ndim == 3 and original_data.shape[-1] in (3, 4):
                            blurred = np.zeros_like(original_data)
                            for channel in range(original_data.shape[-1]):
                                blurred[..., channel] = gaussian_filter(original_data[..., channel], sigma=5)
                            transformed_data = blurred
                        else:
                            transformed_data = gaussian_filter(original_data, sigma=5)
                        transformed_data = transformed_data.astype(np.uint8)
                        self.image_manager.save_image(self.current_image_index, transformed_data)
                        self.viewer.layers[0].data = transformed_data
                        self.logger.info("Applied and saved Gaussian blur to the image.")
                        self.ui_msgbar.setText("Applied and saved Gaussian blur to the image.")
                else:
                    original_data = self.image_manager.get_original_image(self.current_image_index)
                    if original_data is not None:
                        self.image_manager.save_image(self.current_image_index, original_data)
                        self.viewer.layers[0].data = original_data
                        self.logger.info("Reverted and saved original image.")
                        self.ui_msgbar.setText("Reverted and saved original image.")

    def on_image_selected(self, index: int) -> None:
        """Handles image selection from the UI."""
        if 0 <= index < len(self.image_manager.image_files):
            self.save_current_annotation()
            self.current_image_index = index
            self.update_image_and_ui()
