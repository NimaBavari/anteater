import hashlib
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
from qtpy.QtWidgets import QApplication

from annotator import AnnotationManager, ImageAnnotator, ImageManager, UIComponent
from aux import generate_directory_checksum
from constants import CLASS_HIERARCHY

app = None


def setUpModule():
    global app
    app = QApplication([])


def tearDownModule():
    global app
    app.quit()
    app = None


class TestImageManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.error = MagicMock()
        self.logger.info = MagicMock()
        self.temp_dir = TemporaryDirectory()
        self.image_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    def test_init_no_images(self, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_iterdir.return_value = []
        with self.assertRaises(Exception) as cm:
            ImageManager(self.image_dir, self.logger)
        self.assertEqual(str(cm.exception), f"No supported image files found in '{self.image_dir}'.")

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    def test_init_valid_images(self, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file1 = MagicMock(spec=Path, suffix=".png")
        mock_file2 = MagicMock(spec=Path, suffix=".jpg")
        mock_iterdir.return_value = [mock_file1, mock_file2]
        manager = ImageManager(self.image_dir, self.logger)
        self.assertEqual(manager.image_files, [mock_file1, mock_file2])
        self.assertTrue((Path(self.image_dir) / ".originals").exists.called)

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("shutil.copy2")
    def test_backup_originals(self, mock_copy2, mock_exists, mock_iterdir):
        mock_exists.return_value = False
        mock_file1 = MagicMock(spec=Path, suffix=".png")
        mock_file1.name = "test1.png"
        mock_iterdir.return_value = [mock_file1]
        manager = ImageManager(self.image_dir, self.logger)
        mock_copy2.assert_called_with(mock_file1, manager.original_dir / "test1.png")
        self.logger.info.assert_called_with(f"Backed up original image to '{manager.original_dir / 'test1.png'}'.")

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imread")
    def test_get_image_valid(self, mock_imread, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        mock_image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_imread.return_value = mock_image
        result = manager.get_image(0)
        self.assertTrue(np.array_equal(result, mock_image))
        self.assertEqual(result.dtype, np.uint8)
        mock_imread.assert_called_with(mock_file)

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imread")
    def test_get_original_image_valid(self, mock_imread, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        mock_image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_imread.return_value = mock_image
        result = manager.get_original_image(0)
        self.assertTrue(np.array_equal(result, mock_image))
        self.assertEqual(result.dtype, np.uint8)
        mock_imread.assert_called_with(manager.original_dir / "test.png")

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    def test_get_image_invalid_index(self, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        result = manager.get_image(1)
        self.assertIsNone(result)

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imread")
    def test_get_image_load_failure(self, mock_imread, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        mock_imread.side_effect = Exception("Load error")
        result = manager.get_image(0)
        self.assertIsNone(result)
        self.logger.error.assert_called_with(f"Failed to load image '{mock_file}': Load error")

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imread")
    def test_get_original_image_load_failure(self, mock_imread, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        mock_imread.side_effect = Exception("Load error")
        result = manager.get_original_image(0)
        self.assertIsNone(result)
        self.logger.error.assert_called_with(
            f"Failed to load original image '{manager.original_dir / 'test.png'}': Load error"
        )

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    def test_get_image_name_valid(self, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        result = manager.get_image_name(0)
        self.assertEqual(result, "test.png")

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    def test_get_image_name_invalid_index(self, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        result = manager.get_image_name(1)
        self.assertIsNone(result)

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imwrite")
    def test_save_image_valid(self, mock_imwrite, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        image_data = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        manager.save_image(0, image_data)
        calls = mock_imwrite.call_args_list
        self.assertTrue(any(np.array_equal(call[0][1], image_data) and call[0][0] == mock_file for call in calls))
        self.logger.info.assert_called_with(
            f"Saved image to '{mock_file}' with shape {image_data.shape}, dtype {image_data.dtype}."
        )

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.exists")
    @patch("imageio.v3.imwrite")
    def test_save_image_failure(self, mock_imwrite, mock_exists, mock_iterdir):
        mock_exists.return_value = True
        mock_file = MagicMock(spec=Path, suffix=".png")
        mock_file.name = "test.png"
        mock_iterdir.return_value = [mock_file]
        manager = ImageManager(self.image_dir, self.logger)
        image_data = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_imwrite.side_effect = Exception("Save error")
        manager.save_image(0, image_data)
        self.logger.error.assert_called_with(f"Failed to save image to '{mock_file}': Save error")


class TestAnnotationManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.info = MagicMock()
        self.logger.error = MagicMock()
        self.annotator = MagicMock(spec=ImageAnnotator, logger=self.logger)
        self.annotator.ui_msgbar = MagicMock(spec_set=["setText", "setReadOnly"])
        self.temp_dir = TemporaryDirectory()
        self.annotation_dir = Path(self.temp_dir.name)
        self.manager = AnnotationManager(self.annotation_dir, self.annotator)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("pathlib.Path.mkdir")
    def test_init(self, mock_mkdir):
        mock_mkdir.return_value = None
        manager = AnnotationManager(self.annotation_dir, self.annotator)
        self.assertEqual(manager.annotation_dir, Path(self.annotation_dir))
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        self.logger.info.assert_called_with(f"Annotation directory set to '{self.annotation_dir}'.")

    @patch("builtins.open")
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_annotations(self, mock_exists, mock_json_load, mock_open):
        mock_file = MagicMock(spec=Path, name="test.png", stem="test")
        mock_file.name = "test.png"
        mock_exists.return_value = True
        mock_json_load.return_value = {"labels": ["Animal", "Cat", "Persian cat"], "is_transformed": True}
        self.manager.load_annotations([mock_file])
        self.assertEqual(self.manager.annotations, {"test.png": (["Animal", "Cat", "Persian cat"], True)})
        self.logger.info.assert_called_with("Loaded annotation for 'test.png'.")
        self.annotator.ui_msgbar.setText.assert_called_with("Loaded annotation for 'test.png'.")

    @patch("builtins.open")
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_annotations_missing_is_transformed(self, mock_exists, mock_json_load, mock_open):
        mock_file = MagicMock(spec=Path, name="test.png", stem="test")
        mock_file.name = "test.png"
        mock_exists.return_value = True
        mock_json_load.return_value = {"labels": ["Animal", "Cat", "Persian cat"]}
        self.manager.load_annotations([mock_file])
        self.assertEqual(self.manager.annotations, {"test.png": (["Animal", "Cat", "Persian cat"], False)})
        self.logger.info.assert_called_with("Loaded annotation for 'test.png'.")
        self.annotator.ui_msgbar.setText.assert_called_with("Loaded annotation for 'test.png'.")

    @patch("builtins.open")
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_annotations_failure(self, mock_exists, mock_json_load, mock_open):
        mock_file = MagicMock(spec=Path, name="test.png", stem="test")
        mock_exists.return_value = True
        mock_json_load.side_effect = Exception("JSON error")
        self.manager.load_annotations([mock_file])
        self.assertEqual(self.manager.annotations, {})
        self.logger.error.assert_called_with(f"Failed to load annotation '{self.annotation_dir}/test.json': JSON error")
        self.annotator.ui_msgbar.setText.assert_called_with(
            f"Failed to load annotation '{self.annotation_dir}/test.json'."
        )

    @patch("builtins.open")
    @patch("json.dump")
    def test_save_annotation(self, mock_json_dump, mock_open):
        image_name = "test.png"
        annotation = ["Animal", "Cat", "Persian cat"]
        is_transformed = True
        self.manager.save_annotation(image_name, annotation, is_transformed)
        mock_json_dump.assert_called_with(
            {"labels": annotation, "is_transformed": is_transformed}, mock_open().__enter__()
        )
        self.logger.info.assert_called_with(
            f"Saved annotation for '{image_name}': {annotation}, is_transformed: {is_transformed}."
        )
        self.annotator.ui_msgbar.setText.assert_called_with(f"Saved annotation for '{image_name}': {annotation}.")

    @patch("builtins.open")
    @patch("json.dump")
    def test_save_annotation_failure(self, mock_json_dump, mock_open):
        image_name = "test.png"
        annotation = ["Animal", "Cat", "Persian cat"]
        is_transformed = False
        mock_json_dump.side_effect = Exception("Save error")
        self.manager.save_annotation(image_name, annotation, is_transformed)
        self.logger.error.assert_called_with(f"Failed to save annotation for '{image_name}': Save error")
        self.annotator.ui_msgbar.setText.assert_called_with(f"Failed to save annotation for '{image_name}'.")

    def test_get_annotation_existing(self):
        self.manager.annotations = {"test.png": (["Animal", "Cat"], True)}
        result = self.manager.get_annotation("test.png")
        self.assertEqual(result, (["Animal", "Cat"], True))

    def test_get_annotation_non_existing(self):
        result = self.manager.get_annotation("test.png")
        self.assertEqual(result, ([], False))


class TestUIComponent(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.info = MagicMock()
        self.annotator = MagicMock(spec=ImageAnnotator, logger=self.logger)
        mock_file1 = MagicMock(spec=Path)
        mock_file1.name = "test1.png"
        mock_file2 = MagicMock(spec=Path)
        mock_file2.name = "test2.jpg"
        self.annotator.image_manager = MagicMock(image_files=[mock_file1, mock_file2])
        self.annotator.viewer = MagicMock()
        self.annotator.ui_msgbar = MagicMock(spec_set=["setText", "setReadOnly"])
        self.mock_category_combo = MagicMock()
        self.mock_category_combo._value = None
        self.mock_category_combo.choices = ["Animal", "Other"]
        self.mock_category_combo.enabled = True
        self.mock_category_combo.native.currentIndex.return_value = -1
        self.mock_category_combo.value = property(
            lambda self: self._value, lambda self, val: setattr(self, "_value", val)
        )

        self.mock_subcategory_combo = MagicMock()
        self.mock_subcategory_combo._value = None
        self.mock_subcategory_combo.choices = list(CLASS_HIERARCHY["Animal"].keys())
        self.mock_subcategory_combo.enabled = True
        self.mock_subcategory_combo.native.currentIndex.return_value = -1
        self.mock_subcategory_combo.value = property(
            lambda self: self._value, lambda self, val: setattr(self, "_value", val)
        )

        self.mock_breed_combo = MagicMock()
        self.mock_breed_combo._value = None
        self.mock_breed_combo.choices = CLASS_HIERARCHY["Animal"]["Cat"]
        self.mock_breed_combo.enabled = True
        self.mock_breed_combo.native.currentIndex.return_value = -1
        self.mock_breed_combo.value = property(lambda self: self._value, lambda self, val: setattr(self, "_value", val))

        self.mock_transform_checkbox = MagicMock()
        self.mock_transform_checkbox.isChecked.return_value = False

        def combo_box_factory(*args, **kwargs):
            if not hasattr(self, "_combo_mocks"):
                self._combo_mocks = [self.mock_category_combo, self.mock_subcategory_combo, self.mock_breed_combo]
            try:
                return self._combo_mocks.pop(0)
            except IndexError:
                return MagicMock()

        def checkbox_factory(*args, **kwargs):
            return self.mock_transform_checkbox

        with patch("napari.Viewer", autospec=True), patch("qtpy.QtWidgets.QListWidget", autospec=True), patch(
            "magicgui.widgets.Container", autospec=True
        ), patch("magicgui.widgets.ComboBox", side_effect=combo_box_factory), patch(
            "qtpy.QtWidgets.QCheckBox", side_effect=checkbox_factory
        ):
            self.ui = UIComponent(self.annotator)

    def test_init(self):
        self.assertEqual(self.ui.image_names, ["test1.png", "test2.jpg"])
        self.logger.info.assert_called_with("UI components initialized.")
        self.annotator.ui_msgbar.setText.assert_called_with("UI components initialized.")

    def test_on_category_changed_animal(self):
        self.ui.on_category_changed("Animal")
        self.assertEqual(list(self.ui.subcategory_combo.choices), list(CLASS_HIERARCHY["Animal"].keys()))
        self.assertTrue(self.ui.subcategory_combo.enabled)

    def test_on_category_changed_other(self):
        self.ui.on_category_changed("Other")
        self.assertEqual(list(self.ui.subcategory_combo.choices), [])
        self.assertFalse(self.ui.subcategory_combo.enabled)
        self.assertFalse(self.ui.breed_combo.enabled)

    def test_on_subcategory_changed_cat(self):
        self.ui.on_subcategory_changed("Cat")
        self.assertEqual(list(self.ui.breed_combo.choices), CLASS_HIERARCHY["Animal"]["Cat"])
        self.assertTrue(self.ui.breed_combo.enabled)

    def test_on_subcategory_changed_tiger(self):
        self.ui.on_subcategory_changed("Tiger")
        self.assertEqual(list(self.ui.breed_combo.choices), [])
        self.assertFalse(self.ui.breed_combo.enabled)

    def test_update_ui_with_annotation(self):
        annotation = ["Animal", "Cat", "Persian cat"]
        self.ui.update_ui(annotation, True)
        self.assertEqual(self.ui.category_combo.value, "Animal")
        self.assertEqual(self.ui.subcategory_combo.value, "Cat")
        self.assertEqual(self.ui.breed_combo.value, "Persian cat")
        self.logger.info.assert_called_with(f"UI updated with annotation: {annotation}, is_transformed: True.")
        self.annotator.ui_msgbar.setText.assert_called_with(f"UI updated with annotation: {annotation}.")

    def test_get_current_annotation_other(self):
        self.ui.category_combo.value = "Other"
        result = self.ui.get_current_annotation()
        self.assertEqual(result, ["Other"])


class TestImageAnnotator(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test")
        self.logger.info = MagicMock()
        self.logger.error = MagicMock()
        self.temp_dir = TemporaryDirectory()
        self.image_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("annotator.ImageManager")
    def test_init_no_images(self, mock_image_manager):
        mock_image_manager.side_effect = Exception("No images")
        with self.assertRaises(Exception) as cm:
            ImageAnnotator(self.image_dir, self.logger)
        self.assertEqual(str(cm.exception), "No images")

    @patch("annotator.ImageManager")
    @patch("annotator.AnnotationManager")
    @patch("annotator.UIComponent")
    @patch("napari.Viewer")
    def test_init_with_images(self, mock_viewer, mock_ui_component, mock_annotation_manager, mock_image_manager):
        mock_file = MagicMock(name="test.png")
        mock_image_manager.return_value.image_files = [mock_file]
        mock_image_manager.return_value.get_image.return_value = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_image_manager.return_value.get_image_name.return_value = "test.png"
        mock_annotation_manager.return_value.get_annotation.return_value = ([], False)
        annotator = ImageAnnotator(self.image_dir, self.logger)
        self.assertEqual(annotator.current_image_index, 0)
        mock_image_manager.assert_called_with(self.image_dir, self.logger)
        mock_annotation_manager.assert_called()
        mock_ui_component.assert_called_with(annotator)
        mock_viewer.assert_called()

    @patch("annotator.ImageManager")
    @patch("annotator.AnnotationManager")
    @patch("annotator.UIComponent")
    @patch("napari.Viewer")
    @patch("scipy.ndimage.gaussian_filter")
    def test_update_image_and_ui(
        self, mock_gaussian_filter, mock_viewer, mock_ui_component, mock_annotation_manager, mock_image_manager
    ):
        mock_image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_file = MagicMock(name="test.png")
        mock_image_manager.return_value.get_image.return_value = mock_image
        mock_image_manager.return_value.get_image_name.return_value = "test.png"
        mock_image_manager.return_value.image_files = [mock_file]
        mock_annotation_manager.return_value.get_annotation.return_value = (["Animal", "Cat"], True)
        mock_gaussian_filter.return_value = mock_image
        mock_viewer.return_value.layers = []
        annotator = ImageAnnotator(self.image_dir, self.logger)
        annotator.update_image_and_ui()
        mock_viewer.return_value.add_image.call_args_list
        mock_ui_component.return_value.update_ui.assert_called_with(["Animal", "Cat"], True)
        self.logger.info.assert_called_with("Updated image and UI for 'test.png'.")

    @patch("annotator.ImageManager")
    @patch("annotator.AnnotationManager")
    @patch("annotator.UIComponent")
    @patch("napari.Viewer")
    @patch("scipy.ndimage.gaussian_filter")
    def test_save_current_annotation_with_blur(
        self, mock_gaussian_filter, mock_viewer, mock_ui_component, mock_annotation_manager, mock_image_manager
    ):
        mock_image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_file = MagicMock(name="test.png")
        mock_image_manager.return_value.get_image_name.return_value = "test.png"
        mock_image_manager.return_value.get_original_image.return_value = mock_image
        mock_image_manager.return_value.image_files = [mock_file]
        mock_ui_component.return_value.get_current_annotation.return_value = ["Animal", "Cat"]
        mock_ui_component.return_value.transform_checkbox.isChecked.return_value = True
        mock_gaussian_filter.return_value = mock_image
        mock_viewer.return_value.layers = [MagicMock(data=mock_image)]
        mock_annotation_manager.return_value.get_annotation.return_value = ([], False)
        annotator = ImageAnnotator(self.image_dir, self.logger)
        annotator.save_current_annotation()
        mock_annotation_manager.return_value.save_annotation.assert_called_with("test.png", ["Animal", "Cat"], True)
        mock_image_manager.return_value.save_image.assert_called()
        self.logger.info.assert_called_with("Applied and saved Gaussian blur to the image.")

    @patch("annotator.ImageManager")
    @patch("annotator.AnnotationManager")
    @patch("annotator.UIComponent")
    @patch("napari.Viewer")
    def test_save_current_annotation_without_blur(
        self, mock_viewer, mock_ui_component, mock_annotation_manager, mock_image_manager
    ):
        mock_image = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        mock_file = MagicMock(name="test.png")
        mock_image_manager.return_value.get_image_name.return_value = "test.png"
        mock_image_manager.return_value.get_original_image.return_value = mock_image
        mock_image_manager.return_value.image_files = [mock_file]
        mock_ui_component.return_value.get_current_annotation.return_value = ["Animal", "Cat"]
        mock_ui_component.return_value.transform_checkbox.isChecked.return_value = False
        mock_viewer.return_value.layers = [MagicMock(data=mock_image)]
        mock_annotation_manager.return_value.get_annotation.return_value = ([], False)
        annotator = ImageAnnotator(self.image_dir, self.logger)
        annotator.save_current_annotation()
        mock_annotation_manager.return_value.save_annotation.assert_called_with("test.png", ["Animal", "Cat"], False)
        mock_image_manager.return_value.save_image.assert_called()
        self.logger.info.assert_called_with("Reverted and saved original image.")

    @patch("annotator.ImageManager")
    @patch("annotator.AnnotationManager")
    @patch("annotator.UIComponent")
    @patch("napari.Viewer")
    @patch.object(ImageAnnotator, "update_image_and_ui")
    def test_on_image_selected(
        self, mock_update_image_and_ui, mock_viewer, mock_ui_component, mock_annotation_manager, mock_image_manager
    ):
        mock_file = MagicMock(name="test.png")
        mock_image_manager.return_value.image_files = [mock_file]
        mock_image_manager.return_value.get_image_name.return_value = "test.png"
        mock_annotation_manager.return_value.get_annotation.return_value = ([], False)
        annotator = ImageAnnotator(self.image_dir, self.logger)
        annotator.on_image_selected(0)
        mock_annotation_manager.return_value.save_annotation.assert_called()
        self.assertEqual(annotator.current_image_index, 0)
        mock_update_image_and_ui.assert_called()


class TestAuxFunctions(unittest.TestCase):
    @patch("pathlib.Path.iterdir")
    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_generate_directory_checksum_originals(self, mock_exists, mock_open, mock_iterdir):
        mock_exists.side_effect = [True]
        mock_file1 = MagicMock(spec=Path, suffix=".png", is_file=lambda: True)
        mock_file1.name = "image1.png"
        mock_file2 = MagicMock(spec=Path, suffix=".jpg", is_file=lambda: True)
        mock_file2.name = "image2.jpg"
        mock_file1.__lt__ = lambda x, y: x.name < y.name
        mock_file2.__lt__ = lambda x, y: x.name < y.name
        mock_iterdir.return_value = [mock_file2, mock_file1]
        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"data1")))),
            MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"data2")))),
        ]
        result = generate_directory_checksum("/fake/dir")
        expected = hashlib.sha256(
            f"image1.png{hashlib.sha256(b'data1').hexdigest()}image2.jpg{hashlib.sha256(b'data2').hexdigest()}".encode()
        ).hexdigest()
        self.assertEqual(result, expected)

    @patch("pathlib.Path.iterdir")
    @patch("builtins.open")
    @patch("pathlib.Path.exists")
    def test_generate_directory_checksum_no_originals(self, mock_exists, mock_open, mock_iterdir):
        mock_exists.side_effect = [False]
        mock_file1 = MagicMock(spec=Path, suffix=".png", is_file=lambda: True)
        mock_file1.name = "image1.png"
        mock_file2 = MagicMock(spec=Path, suffix=".jpg", is_file=lambda: True)
        mock_file2.name = "image2.jpg"
        mock_file1.__lt__ = lambda x, y: x.name < y.name
        mock_file2.__lt__ = lambda x, y: x.name < y.name
        mock_iterdir.return_value = [mock_file2, mock_file1]
        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"data1")))),
            MagicMock(__enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=b"data2")))),
        ]
        result = generate_directory_checksum("/fake/dir")
        expected = hashlib.sha256(
            f"image1.png{hashlib.sha256(b'data1').hexdigest()}image2.jpg{hashlib.sha256(b'data2').hexdigest()}".encode()
        ).hexdigest()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
