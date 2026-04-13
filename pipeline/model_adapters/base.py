"""Base class for multimodal model adapters."""

import abc
import logging

logger = logging.getLogger(__name__)


class BaseModelAdapter(abc.ABC):
    """Abstract adapter that wraps a multimodal VLM for inference.

    Subclasses must implement :meth:`generate`.  The default context-manager
    protocol calls :meth:`_load_model` on entry and :meth:`_unload_model`
    on exit so weights can be freed after the generation loop.

    Usage::

        adapter = get_adapter("llava", device="cuda:0")
        with adapter:
            generation = adapter.generate(text="...", image_path="/abs/img.jpg")
    """

    def __init__(self, config: str | None = None, device: str = "cuda:0", max_new_tokens: int = 512):
        """
        Args:
            config: Optional path to a YAML/JSON config file for the adapter.
            device: PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
            max_new_tokens: Maximum number of new tokens to generate.
        """
        self.config = config
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self):
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._unload_model()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load model weights.  Override in subclasses."""

    def _unload_model(self):
        """Free model weights from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    @abc.abstractmethod
    def generate(self, text: str, image_path: str) -> str:
        """Generate a text response for the given prompt and image.

        Args:
            text: The text prompt.
            image_path: Absolute path to the image file.

        Returns:
            The model's text generation (str).
        """
