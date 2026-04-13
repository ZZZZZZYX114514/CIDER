#!/usr/bin/env python3
"""Lightweight dry-run tests for the multimodal pipeline.

Tests run without any deep-learning model weights.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Ensure pipeline/ is on the path
_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "pipeline")
sys.path.insert(0, _PIPELINE_DIR)


# ---------------------------------------------------------------------------
# Stage-1 tests
# ---------------------------------------------------------------------------

class TestValidateTestCase(unittest.TestCase):
    """Unit tests for generate_test_cases.validate_test_case."""

    @classmethod
    def setUpClass(cls):
        from generate_test_cases import validate_test_case
        cls.validate = staticmethod(validate_test_case)
        # Temporary directory with a dummy image
        cls.tmp = tempfile.mkdtemp()
        cls.images_dir = os.path.join(cls.tmp, "images")
        os.makedirs(cls.images_dir)
        # Create a tiny dummy image file
        cls.dummy_img = os.path.join(cls.images_dir, "test.jpg")
        with open(cls.dummy_img, "wb") as fh:
            fh.write(b"\xff\xd8\xff")  # minimal JPEG magic bytes

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_valid_test_case(self):
        tc = {"text": "hello", "image": "images/test.jpg"}
        self.validate(tc, self.tmp)  # should not raise

    def test_missing_text(self):
        tc = {"image": "images/test.jpg"}
        with self.assertRaises(ValueError):
            self.validate(tc, self.tmp)

    def test_missing_image(self):
        tc = {"text": "hello"}
        with self.assertRaises(ValueError):
            self.validate(tc, self.tmp)

    def test_absolute_image_rejected(self):
        tc = {"text": "hello", "image": "/absolute/path/img.jpg"}
        with self.assertRaises(ValueError):
            self.validate(tc, self.tmp)

    def test_missing_file_raises(self):
        tc = {"text": "hello", "image": "images/nonexistent.jpg"}
        with self.assertRaises(FileNotFoundError):
            self.validate(tc, self.tmp)


class TestLoadBehaviors(unittest.TestCase):
    """Unit tests for generate_test_cases.load_behaviors."""

    @classmethod
    def setUpClass(cls):
        from generate_test_cases import load_behaviors
        cls.load = staticmethod(load_behaviors)

    def _write_csv(self, content: str) -> str:
        fh = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        fh.write(content)
        fh.close()
        return fh.name

    def test_headered_csv(self):
        path = self._write_csv(
            "BehaviorID,Behavior,Tags\n"
            "bid1,Do something harmful,standard\n"
            "bid2,Another harmful thing,standard\n"
        )
        try:
            behaviors = self.load(path)
        finally:
            os.unlink(path)
        self.assertIn("bid1", behaviors)
        self.assertEqual(behaviors["bid1"]["Behavior"], "Do something harmful")

    def test_legacy_csv_no_header(self):
        # Legacy format: text=col[0], behavior_id=col[-1]
        path = self._write_csv(
            "Do something harmful,standard,category,bid_legacy\n"
        )
        try:
            behaviors = self.load(path)
        finally:
            os.unlink(path)
        self.assertIn("bid_legacy", behaviors)
        self.assertEqual(behaviors["bid_legacy"]["Behavior"], "Do something harmful")


# ---------------------------------------------------------------------------
# Stage-2 tests (mock adapter – no GPU required)
# ---------------------------------------------------------------------------

class TestGenerateCompletions(unittest.TestCase):
    """Integration test for generate_completions.main using a stub adapter."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        images_dir = os.path.join(self.tmp, "images")
        os.makedirs(images_dir)

        # Create a minimal dummy image
        dummy_img = os.path.join(images_dir, "dummy.jpg")
        with open(dummy_img, "wb") as fh:
            fh.write(b"\xff\xd8\xff")

        # Write a test_cases.json
        test_cases = {
            "behavior_1": [
                {
                    "text": "Test prompt",
                    "image": "images/dummy.jpg",
                    "meta": {"behavior_id": "behavior_1"},
                }
            ]
        }
        tc_path = os.path.join(self.tmp, "test_cases.json")
        with open(tc_path, "w") as fh:
            json.dump(test_cases, fh)
        self.tc_path = tc_path
        self.save_path = os.path.join(self.tmp, "completions.json")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_completions_structure(self):
        """generate_completions produces well-formed completions.json."""
        import model_adapters
        from model_adapters.base import BaseModelAdapter

        # Stub adapter that returns a fixed string
        class StubAdapter(BaseModelAdapter):
            def generate(self, text, image_path):
                return f"stub response for: {text[:20]}"

        # Patch get_adapter
        original_get = model_adapters.get_adapter

        def mock_get_adapter(name, **kwargs):
            return StubAdapter(device="cpu")

        model_adapters.get_adapter = mock_get_adapter
        try:
            # Directly import and call main
            import generate_completions
            generate_completions.main([
                "--model", "llava",
                "--test_cases_path", self.tc_path,
                "--save_path", self.save_path,
            ])
        finally:
            model_adapters.get_adapter = original_get

        self.assertTrue(os.path.isfile(self.save_path))
        with open(self.save_path) as fh:
            completions = json.load(fh)

        self.assertIn("behavior_1", completions)
        entries = completions["behavior_1"]
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertIn("test_case", entry)
        self.assertIn("generation", entry)
        self.assertEqual(entry["test_case"]["text"], "Test prompt")
        self.assertIn("stub response", entry["generation"])


# ---------------------------------------------------------------------------
# Stage-1 integration: image_hijack method
# ---------------------------------------------------------------------------

class TestImageHijackMethod(unittest.TestCase):
    """Integration test for the image_hijack attack method."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Create dummy source images
        self.img_dir = os.path.join(self.tmp, "src_images")
        os.makedirs(self.img_dir)
        for name in ["adv_img_1.jpg", "adv_img_2.jpg"]:
            with open(os.path.join(self.img_dir, name), "wb") as fh:
                fh.write(b"\xff\xd8\xff")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_generate_test_cases(self):
        import yaml
        # Write a minimal config
        cfg_path = os.path.join(self.tmp, "cfg.yaml")
        with open(cfg_path, "w") as fh:
            yaml.safe_dump({"img_dir": self.img_dir, "n_per_behavior": 1}, fh)

        from methods.image_hijack import AttackMethod
        method = AttackMethod(config=cfg_path, save_dir=self.tmp)
        behaviors = {
            "bid1": {"BehaviorID": "bid1", "Behavior": "Harmful prompt 1"},
            "bid2": {"BehaviorID": "bid2", "Behavior": "Harmful prompt 2"},
        }
        test_cases = method.generate_test_cases(behaviors)

        self.assertIn("bid1", test_cases)
        self.assertIn("bid2", test_cases)

        for bid, tcs in test_cases.items():
            self.assertEqual(len(tcs), 1)
            tc = tcs[0]
            self.assertIn("text", tc)
            self.assertIn("image", tc)
            self.assertFalse(os.path.isabs(tc["image"]))
            self.assertTrue(tc["image"].startswith("images/"))
            img_abs = os.path.join(self.tmp, tc["image"])
            self.assertTrue(os.path.isfile(img_abs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
