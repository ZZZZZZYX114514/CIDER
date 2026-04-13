"""Qwen-VL adapter.

Supports both the original Qwen-VL-Chat (``qwen_path`` in settings) and
the newer Qwen2-VL (``Qwen2_VL_7B`` in settings).  The variant is
selected by inspecting the model path.

Reference: code/utils.py::get_response (qwen branch).
"""

import logging
import os

import yaml

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SETTINGS_PATH = os.path.join(_REPO_ROOT, "settings", "settings.yaml")

with open(_SETTINGS_PATH) as _fh:
    _settings = yaml.safe_load(_fh)


class QwenVLAdapter(BaseModelAdapter):
    """Adapter for Qwen-VL-Chat and Qwen2-VL models."""

    def __init__(self, use_qwen2: bool = False, **kwargs):
        """
        Args:
            use_qwen2: When ``True`` load Qwen2-VL (``Qwen2_VL_7B`` key);
                       otherwise load Qwen-VL-Chat (``qwen_path`` key).
        """
        super().__init__(**kwargs)
        self.use_qwen2 = use_qwen2
        self._tokenizer = None
        self._processor = None

    # ------------------------------------------------------------------

    def _load_model(self):
        import torch

        if self.use_qwen2:
            from transformers import (
                AutoProcessor,
                Qwen2VLForConditionalGeneration,
            )
            model_path = _settings["Qwen2_VL_7B"]
            logger.info("Loading Qwen2-VL from %s", model_path)
            self._processor = AutoProcessor.from_pretrained(model_path)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_path = _settings["qwen_path"]
            logger.info("Loading Qwen-VL-Chat from %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, fp32=True
            ).eval().to(self.device)

    def _unload_model(self):
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        super()._unload_model()

    # ------------------------------------------------------------------

    def generate(self, text: str, image_path: str) -> str:
        """Generate a response using Qwen-VL.

        Args:
            text: Text prompt.
            image_path: Absolute path to the image.

        Returns:
            Model's text generation.
        """
        if self._model is None:
            self._load_model()

        if self.use_qwen2:
            return self._generate_qwen2(text, image_path)
        return self._generate_qwen_chat(text, image_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_qwen_chat(self, text: str, image_path: str) -> str:
        query = self._tokenizer.from_list_format(  # type: ignore[attr-defined]
            [{"image": image_path}, {"text": text}]
        )
        answer, _history = self._model.chat(
            self._tokenizer,
            query=query,
            history=None,
            max_new_tokens=self.max_new_tokens,
        )
        return answer

    def _generate_qwen2(self, text: str, image_path: str) -> str:
        from qwen_vl_utils import process_vision_info
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens
        )
        # Strip the input tokens from the output
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
