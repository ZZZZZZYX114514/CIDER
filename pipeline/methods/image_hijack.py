"""ImageHijack method: pairs a harmful-text behavior with an adversarial image.

This is a *reference* implementation that demonstrates how a method should
be structured.  It reads adversarial (or clean) images from a local
directory and produces one text-image test case per behavior.

Config keys (YAML file passed via ``--method_config``):
    img_dir  (str): Directory containing source images.  Defaults to
                    ``data/img/adv4`` relative to the repo root.
    n_per_behavior (int): How many test cases to generate per behavior.
                          Defaults to 1.
"""

import os
import shutil

from .base import BaseMethod

# Resolve paths relative to repo root (two levels above this file)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_IMG_DIR = os.path.join(_REPO_ROOT, "data", "img", "adv4")


class AttackMethod(BaseMethod):
    """Simple image-hijack method.

    For each behavior it selects an adversarial image (cycling through the
    available images) and pairs it with the raw behavior text.
    """

    def __init__(self, config=None, save_dir: str = "."):
        super().__init__(config=config, save_dir=save_dir)
        self.img_dir = self._cfg.get("img_dir", _DEFAULT_IMG_DIR)
        self.n_per_behavior = int(self._cfg.get("n_per_behavior", 1))

    # ------------------------------------------------------------------

    def generate_test_cases(self, behaviors: dict) -> dict:
        """Generate test cases by pairing each behavior with adversarial images.

        Args:
            behaviors: ``{behavior_id: {BehaviorID, Behavior, ...}}``

        Returns:
            ``{behavior_id: [test_case_dict, ...]}``
        """
        img_files = sorted(
            f
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))
        )
        if not img_files:
            raise FileNotFoundError(
                f"No image files found in img_dir: {self.img_dir}"
            )

        test_cases: dict = {}
        img_count = len(img_files)

        for idx, (bid, behavior) in enumerate(behaviors.items()):
            test_cases[bid] = []
            for k in range(self.n_per_behavior):
                img_file = img_files[(idx * self.n_per_behavior + k) % img_count]
                src_img = os.path.join(self.img_dir, img_file)

                # Copy image to images/ dir under save_dir
                safe_bid = bid.replace("/", "_").replace("\\", "_")
                dst_fname = f"{safe_bid}_{k}_{img_file}"
                dst_abs = os.path.join(self._images_dir, dst_fname)
                shutil.copy2(src_img, dst_abs)

                test_case = {
                    "text": behavior["Behavior"],
                    "image": os.path.join("images", dst_fname),
                    "image_source": src_img,
                    "meta": {
                        "behavior_id": bid,
                        "method": "image_hijack",
                    },
                }
                test_cases[bid].append(test_case)

        return test_cases
