"""Model adapter registry for completion generation."""

from .base import BaseModelAdapter


def get_adapter(model_name: str, **kwargs) -> BaseModelAdapter:
    """Return an initialised model adapter.

    Args:
        model_name: One of ``llava``, ``llava1.6``, ``qwen_vl``,
                    ``minigpt4``, ``openai_vision``.
        **kwargs: Forwarded to the adapter constructor
                  (``device``, ``max_new_tokens``, etc.).

    Returns:
        A :class:`BaseModelAdapter` instance that has *not* yet loaded
        the model weights.  Use it as a context manager or call
        :py:meth:`~BaseModelAdapter._load_model` explicitly.
    """
    name = model_name.lower()
    if name in ("llava", "llava1.6"):
        from .llava import LLaVAAdapter
        return LLaVAAdapter(model_name=name, **kwargs)
    if name == "qwen_vl":
        from .qwen_vl import QwenVLAdapter
        return QwenVLAdapter(**kwargs)
    if name == "minigpt4":
        from .minigpt4 import MiniGPT4Adapter
        return MiniGPT4Adapter(**kwargs)
    if name == "openai_vision":
        from .openai_vision import OpenAIVisionAdapter
        return OpenAIVisionAdapter(**kwargs)
    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: "
        "llava, llava1.6, qwen_vl, minigpt4, openai_vision"
    )


__all__ = ["get_adapter", "BaseModelAdapter"]
