"""OpenAI Vision adapter (GPT-4V / GPT-4o).

API credentials are read from ``openai_env_var.json`` in the repo root
(same convention as code/utils.py) or from environment variables
``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``.

Reference: code/utils.py::get_response (gpt4 branch) and encode_image.
"""

import base64
import json
import logging
import os
import time
from io import BytesIO

from .base import BaseModelAdapter

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OPENAI_ENV_FILE = os.path.join(_REPO_ROOT, "openai_env_var.json")


def _encode_image(image_path: str) -> str:
    """Convert an image file to a base64-encoded JPEG string."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OpenAIVisionAdapter(BaseModelAdapter):
    """Adapter for OpenAI vision-capable models (GPT-4V, GPT-4o, etc.)."""

    def __init__(
        self,
        model_id: str = "gpt-4-vision-preview",
        max_retries: int = 6,
        retry_wait: float = 3.0,
        **kwargs,
    ):
        """
        Args:
            model_id: OpenAI model identifier.
            max_retries: Maximum number of retry attempts on API errors.
            retry_wait: Seconds to wait between retries.
        """
        super().__init__(**kwargs)
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self._client = None

    # ------------------------------------------------------------------

    def _load_model(self):
        """Initialise the OpenAI client."""
        import httpx
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")

        # Fall back to the JSON credentials file used by code/utils.py
        if not api_key and os.path.exists(_OPENAI_ENV_FILE):
            with open(_OPENAI_ENV_FILE) as fh:
                creds = json.load(fh)
            api_key = creds.get("api_key", "")
            base_url = base_url or creds.get("url", "")

        if not api_key:
            raise EnvironmentError(
                "OpenAI API key not found.  Set OPENAI_API_KEY or create "
                f"{_OPENAI_ENV_FILE} with {{\"api_key\": \"...\", \"url\": \"...\"}}."
            )

        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["http_client"] = httpx.Client(
                base_url=base_url, follow_redirects=True
            )
        self._client = OpenAI(**client_kwargs)
        logger.info("OpenAI Vision client initialised (model: %s).", self.model_id)

    def _unload_model(self):
        if self._client is not None:
            del self._client
            self._client = None

    # ------------------------------------------------------------------

    def generate(self, text: str, image_path: str) -> str:
        """Generate a response via the OpenAI Chat Completions API.

        Args:
            text: Text prompt.
            image_path: Absolute path to the image.

        Returns:
            Model's text generation.
        """
        if self._client is None:
            self._load_model()

        b64 = _encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "auto",
                        },
                    },
                ],
            }
        ]

        answer = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                )
                answer = response.choices[0].message.content or ""
                break
            except Exception as exc:
                if attempt == self.max_retries:
                    logger.error("OpenAI API error after %d retries: %s", attempt, exc)
                    break
                logger.warning("OpenAI API error (attempt %d): %s", attempt, exc)
                time.sleep(self.retry_wait)

        time.sleep(1)  # rate-limit courtesy pause
        return answer
