"""MiniGPT-4 adapter.

Wraps the Chat / conversation helper that ships with MiniGPT-4.

Reference: code/utils.py::get_response (minigpt4 branch) and the
helper functions query_minigpt / upload_img / ask / answer.
"""

import logging
import os
import sys

import yaml

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SETTINGS_PATH = os.path.join(_REPO_ROOT, "settings", "settings.yaml")

with open(_SETTINGS_PATH) as _fh:
    _settings = yaml.safe_load(_fh)

# MiniGPT-4 helpers live in code/ – add to path when adapter is used
_CODE_DIR = os.path.join(_REPO_ROOT, "code")


class MiniGPT4Adapter(BaseModelAdapter):
    """Adapter for MiniGPT-4.

    Requires the MiniGPT-4 source tree under ``code/models/minigpt4``
    and the eval config at ``code/models/minigpt4/minigpt4_eval.yaml``.
    """

    _DEFAULT_CFG = os.path.join(_CODE_DIR, "models", "minigpt4", "minigpt4_eval.yaml")

    def __init__(self, minigpt4_cfg: str | None = None, **kwargs):
        """
        Args:
            minigpt4_cfg: Path to ``minigpt4_eval.yaml``.  Defaults to
                          ``code/models/minigpt4/minigpt4_eval.yaml``.
        """
        super().__init__(**kwargs)
        self.minigpt4_cfg = minigpt4_cfg or self._DEFAULT_CFG
        self._chat = None
        self._vis_processor = None

    # ------------------------------------------------------------------

    def _load_model(self):
        import types
        import torch

        if _CODE_DIR not in sys.path:
            sys.path.insert(0, _CODE_DIR)

        from models.minigpt4.common.config import Config
        from models.minigpt4.common.registry import registry
        from models.minigpt4.conversation.conversation import (
            Chat,
            CONV_VISION_Vicuna0,
        )

        device_int = int(self.device.split(":")[-1]) if ":" in self.device else 0
        args_ns = types.SimpleNamespace(cfg_path=self.minigpt4_cfg, options=None)
        cfg = Config(args_ns)

        model_config = cfg.model_cfg
        model_config.device_8bit = device_int
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(self.device)

        vis_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)

        self._chat = Chat(model, vis_processor, device=self.device)
        self._model = model  # keep reference for _unload_model
        self._vis_processor = vis_processor
        logger.info("MiniGPT-4 loaded.")

    def _unload_model(self):
        if self._chat is not None:
            del self._chat
            self._chat = None
        if self._vis_processor is not None:
            del self._vis_processor
            self._vis_processor = None
        super()._unload_model()

    # ------------------------------------------------------------------

    def generate(self, text: str, image_path: str) -> str:
        """Generate a response using MiniGPT-4.

        Args:
            text: Text prompt.
            image_path: Absolute path to the image.

        Returns:
            Model's text generation.
        """
        import torch
        from PIL import Image

        if _CODE_DIR not in sys.path:
            sys.path.insert(0, _CODE_DIR)

        from models.minigpt4.conversation.conversation import CONV_VISION_Vicuna0

        if self._chat is None:
            self._load_model()

        img = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            chat_state = CONV_VISION_Vicuna0.copy()
            img_list: list = []
            self._chat.upload_img(img, chat_state, img_list)
            self._chat.encode_img(img_list)
            self._chat.ask(f"<image>\n{text}", chat_state)
            answer, _state, _img_list = self._chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=1,
                temperature=1.0,
                max_new_tokens=self.max_new_tokens,
                max_length=2000,
            )
        return answer
