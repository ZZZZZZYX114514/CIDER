#!/usr/bin/env python3
"""Generate multimodal test cases for attack evaluation.

Three-stage HarmBench-style pipeline – Stage 1.

Usage::

    python generate_test_cases.py \\
        --method image_hijack \\
        --behaviors_file pipeline/behaviors/harmbench_behaviors_multimodal.csv \\
        --save_dir runs/my_run \\
        [--method_config pipeline/methods/configs/image_hijack.yaml]

Output
------
``{save_dir}/test_cases.json``
    JSON manifest with structure ``{behavior_id: [test_case, ...], ...}``.

    Each *test_case* is a dict with at least:

    * ``text``  – text prompt (str)
    * ``image`` – relative path to the image, **must** start with
      ``images/`` and **must not** be absolute.

``{save_dir}/images/``
    Directory containing all referenced image files.
"""

import argparse
import csv
import json
import logging
import os
import sys

# Allow imports from pipeline/methods/ without installing as a package
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PIPELINE_DIR)

from methods import load_method  # noqa: E402  (after sys.path tweak)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_test_case(tc: dict, save_dir: str) -> None:
    """Validate a single test-case dict against the required schema.

    Raises:
        ValueError: If a required field is missing or invalid.
        FileNotFoundError: If the referenced image file does not exist on
            disk after the method has run.
    """
    if "text" not in tc:
        raise ValueError(f"test_case missing required field 'text': {tc!r}")
    if "image" not in tc:
        raise ValueError(f"test_case missing required field 'image': {tc!r}")
    if os.path.isabs(tc["image"]):
        raise ValueError(
            f"'image' must be a relative path, got absolute: {tc['image']!r}"
        )
    img_abs = os.path.join(save_dir, tc["image"])
    if not os.path.isfile(img_abs):
        raise FileNotFoundError(
            f"Image file referenced by test_case does not exist: {img_abs}"
        )


# ---------------------------------------------------------------------------
# Behaviors loading
# ---------------------------------------------------------------------------

def load_behaviors(behaviors_file: str) -> dict:
    """Load behaviors from a CSV file.

    Supported formats:

    1. **Headered CSV** (recommended) – first row contains column names;
       must include ``BehaviorID`` and ``Behavior`` columns.
    2. **HarmBench-legacy CSV** (no header) – columns are:
       ``text, type, category, ..., behavior_id`` (last column).

    Args:
        behaviors_file: Path to the CSV file.

    Returns:
        ``{behavior_id: {BehaviorID, Behavior, ...}}``
    """
    behaviors: dict = {}
    with open(behaviors_file, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        first_row = next(reader, None)
        if first_row is None:
            return behaviors

        # Detect headered format: first row must contain BehaviorID and Behavior
        stripped = [c.strip() for c in first_row]
        if "BehaviorID" in stripped and "Behavior" in stripped:
            col = {name: i for i, name in enumerate(stripped)}
            for row in reader:
                if not row:
                    continue
                bid = row[col["BehaviorID"]].strip()
                behaviors[bid] = {
                    name: row[i] for name, i in col.items() if i < len(row)
                }
        else:
            # Legacy format: text=col[0], behavior_id=col[-1]
            for row in [first_row] + list(reader):
                if not row:
                    continue
                bid = row[-1].strip()
                behaviors[bid] = {"BehaviorID": bid, "Behavior": row[0].strip()}

    logger.info("Loaded %d behaviors from %s", len(behaviors), behaviors_file)
    return behaviors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 1: generate multimodal test cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        required=True,
        help="Name of the attack method module under pipeline/methods/.",
    )
    parser.add_argument(
        "--behaviors_file",
        required=True,
        help="Path to the behaviors CSV file.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory where test_cases.json and images/ will be written.",
    )
    parser.add_argument(
        "--method_config",
        default=None,
        help="Optional YAML config file forwarded to the method.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "images"), exist_ok=True)

    # Load behaviors
    behaviors = load_behaviors(args.behaviors_file)

    # Instantiate attack method
    logger.info("Loading method '%s'", args.method)
    method = load_method(
        args.method, config=args.method_config, save_dir=args.save_dir
    )

    # Generate test cases
    logger.info("Generating test cases…")
    test_cases = method.generate_test_cases(behaviors)

    # Let the method handle image management (copy/symlink externals)
    test_cases = method.save_test_cases(test_cases)

    # Schema validation
    errors: list = []
    for bid, tcs in test_cases.items():
        for tc in tcs:
            try:
                validate_test_case(tc, args.save_dir)
            except (ValueError, FileNotFoundError) as exc:
                errors.append(f"[{bid}] {exc}")

    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        sys.exit(1)

    # Save manifest
    out_path = os.path.join(args.save_dir, "test_cases.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(test_cases, fh, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in test_cases.values())
    logger.info(
        "Saved %d test case(s) across %d behavior(s) to %s",
        total,
        len(test_cases),
        out_path,
    )


if __name__ == "__main__":
    main()
