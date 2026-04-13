"""LLaVA adapter (versions 1.5 and 1.6).

Model paths are read from ``settings/settings.yaml`` using the keys
``llava15_7b_hf_path`` (for ``model_name="llava"``) and
``llava16_7b_hf_path`` (for ``model_name="llava1.6"``).

Reference: code/utils.py::get_response (llava / llava1.6 branches).
"""

import logging
import os
import sys

import yaml

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

# Resolve settings relative to repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SETTINGS_PATH = os.path.join(_REPO_ROOT, "settings", "settings.yaml")

with open(_SETTINGS_PATH) as _fh:
    _settings = yaml.safe_load(_fh)


class LLaVAAdapter(BaseModelAdapter):
    """Adapter for LLaVA-1.5 (``model_name="llava"``) and
    LLaVA-1.6 / LLaVA-NeXT (``model_name="llava1.6"``).
    """

    def __init__(self, model_name: str = "llava", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name.lower()
        self._processor = None

    # ------------------------------------------------------------------

    def _load_model(self):
        import torch
        from transformers import AutoProcessor, AutoModelForPreTraining

        if self.model_name == "llava":
            model_path = _settings["llava15_7b_hf_path"]
            logger.info("Loading LLaVA-1.5 from %s", model_path)
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = AutoModelForPreTraining.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        elif self.model_name == "llava1.6":
            model_path = _settings["llava16_7b_hf_path"]
            logger.info("Loading LLaVA-1.6 from %s", model_path)
            from transformers import (
                LlavaNextProcessor,
                LlavaNextForConditionalGeneration,
            )
            self._processor = LlavaNextProcessor.from_pretrained(model_path)
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported LLaVA variant: {self.model_name!r}")

    def _unload_model(self):
        if self._processor is not None:
            del self._processor
            self._processor = None
        super()._unload_model()

    # ------------------------------------------------------------------

    def generate(self, text: str, image_path: str) -> str:
        """Generate a response using LLaVA.

        Args:
            text: Text prompt.
            image_path: Absolute path to the image.

        Returns:
            Model's text generation.
        """
        from PIL import Image

        if self._model is None:
            self._load_model()

        img = Image.open(image_path).convert("RGB")
        prompt = f"<image>\n{text}\n"
        inputs = self._processor(
            text=prompt,
            images=img,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self._processor.decode(output_ids[0].cpu(), skip_special_tokens=True)
        # Strip the echoed prompt from the decoded output
        return decoded.replace(f"\n{text}\n", "").strip()
