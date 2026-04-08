#!/usr/bin/env python3
"""Generate completions for multimodal test cases.

Three-stage HarmBench-style pipeline – Stage 2.

Usage::

    python generate_completions.py \\
        --model llava \\
        --test_cases_path runs/my_run/test_cases.json \\
        --save_path runs/my_run/completions.json \\
        [--device cuda:0] \\
        [--max_new_tokens 512]

Supported models
----------------
* ``llava``        – LLaVA-1.5 (HuggingFace)
* ``llava1.6``     – LLaVA-1.6 / LLaVA-NeXT (HuggingFace)
* ``qwen_vl``      – Qwen-VL-Chat
* ``minigpt4``     – MiniGPT-4
* ``openai_vision``– GPT-4V / GPT-4o via OpenAI API

Input
-----
``test_cases_path``
    Path to ``test_cases.json`` produced by ``generate_test_cases.py``.
    Structure: ``{behavior_id: [test_case_dict, ...], ...}``

    Each *test_case_dict* must have:

    * ``text``  – text prompt
    * ``image`` – relative path (from ``test_cases.json``'s parent directory)

Output
------
``save_path`` (``completions.json``)
    Structure::

        {
          "<behavior_id>": [
            {
              "test_case": { "text": "...", "image": "images/...", ... },
              "generation": "<model output>"
            },
            ...
          ],
          ...
        }

    This format is compatible with HarmBench-style evaluation scripts.
"""

import argparse
import json
import logging
import os
import sys

# Allow imports from pipeline/model_adapters/ without installing as a package
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PIPELINE_DIR)

from model_adapters import get_adapter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 2: generate completions from test cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["llava", "llava1.6", "qwen_vl", "minigpt4", "openai_vision"],
        help="Target multimodal model.",
    )
    parser.add_argument(
        "--test_cases_path",
        required=True,
        help="Path to test_cases.json produced by generate_test_cases.py.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Output path for completions.json.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device string.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate.",
    )
    # Qwen2-VL flag
    parser.add_argument(
        "--use_qwen2",
        action="store_true",
        help="(qwen_vl only) Use Qwen2-VL instead of Qwen-VL-Chat.",
    )
    # OpenAI flags
    parser.add_argument(
        "--openai_model_id",
        default="gpt-4-vision-preview",
        help="(openai_vision only) OpenAI model identifier.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # -----------------------------------------------------------------------
    # Load test cases
    # -----------------------------------------------------------------------
    test_cases_path = os.path.abspath(args.test_cases_path)
    if not os.path.isfile(test_cases_path):
        logger.error("test_cases_path not found: %s", test_cases_path)
        sys.exit(1)

    with open(test_cases_path, encoding="utf-8") as fh:
        test_cases: dict = json.load(fh)

    test_cases_dir = os.path.dirname(test_cases_path)
    logger.info(
        "Loaded %d behavior(s) from %s",
        len(test_cases),
        test_cases_path,
    )

    # -----------------------------------------------------------------------
    # Build adapter kwargs
    # -----------------------------------------------------------------------
    adapter_kwargs: dict = {
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.model == "qwen_vl":
        adapter_kwargs["use_qwen2"] = args.use_qwen2
    if args.model == "openai_vision":
        adapter_kwargs["model_id"] = args.openai_model_id

    # -----------------------------------------------------------------------
    # Generate completions
    # -----------------------------------------------------------------------
    completions: dict = {}

    logger.info("Loading model adapter for '%s'…", args.model)
    with get_adapter(args.model, **adapter_kwargs) as adapter:
        for bid, tcs in test_cases.items():
            completions[bid] = []
            for tc in tcs:
                # Resolve image path to absolute
                img_rel = tc.get("image", "")
                img_abs = (
                    img_rel
                    if os.path.isabs(img_rel)
                    else os.path.join(test_cases_dir, img_rel)
                )
                img_abs = os.path.abspath(img_abs)

                if not os.path.isfile(img_abs):
                    logger.warning(
                        "[%s] Image not found, skipping: %s", bid, img_abs
                    )
                    generation = ""
                else:
                    try:
                        generation = adapter.generate(
                            text=tc["text"], image_path=img_abs
                        )
                    except Exception as exc:
                        logger.error(
                            "[%s] Generation failed for image %s: %s",
                            bid,
                            img_abs,
                            exc,
                        )
                        generation = ""

                completions[bid].append(
                    {"test_case": tc, "generation": generation}
                )

            logger.info(
                "[%s] Generated %d completion(s).", bid, len(completions[bid])
            )

    # -----------------------------------------------------------------------
    # Save completions
    # -----------------------------------------------------------------------
    save_path = os.path.abspath(args.save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(completions, fh, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in completions.values())
    logger.info("Saved %d completion(s) to %s", total, save_path)


if __name__ == "__main__":
    main()
