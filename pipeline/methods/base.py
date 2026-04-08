"""Base class for multimodal attack methods that generate test cases."""

import abc
import os
import shutil


class BaseMethod(abc.ABC):
    """Abstract base class for test case generation methods.

    Subclasses must implement :meth:`generate_test_cases`.  The default
    :meth:`save_test_cases` copies / symlinks all referenced images into
    ``{save_dir}/images/`` and returns the (possibly rewritten) test-case
    dict so that every ``image`` value starts with ``images/``.
    """

    def __init__(self, config=None, save_dir: str = "."):
        """
        Args:
            config: Optional path to a YAML config file for the method.
            save_dir: Directory where ``test_cases.json`` and ``images/``
                      will be written.
        """
        self.config = config
        self.save_dir = save_dir
        self._images_dir = os.path.join(save_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)

        if config is not None:
            import yaml
            with open(config) as fh:
                self._cfg = yaml.safe_load(fh) or {}
        else:
            self._cfg = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def generate_test_cases(self, behaviors: dict) -> dict:
        """Generate test cases for each behavior.

        Args:
            behaviors: ``{behavior_id: {BehaviorID, Behavior, Tags, ...}}``

        Returns:
            ``{behavior_id: [test_case_dict, ...]}``

            Each *test_case_dict* must contain at least:

            * ``text``  – the text prompt (str)
            * ``image`` – relative path to the image **starting with**
              ``images/`` (str, must NOT be absolute)
        """

    def save_test_cases(self, test_cases: dict) -> dict:
        """Ensure every image is present under ``{save_dir}/images/``.

        Images that already live inside ``{save_dir}/images/`` are left
        untouched.  External images are *copied* into the directory and
        the ``image`` field is updated to the new relative path.

        Args:
            test_cases: mapping returned by :meth:`generate_test_cases`

        Returns:
            Possibly rewritten *test_cases* dict (same structure).
        """
        updated: dict = {}
        for bid, tcs in test_cases.items():
            updated[bid] = []
            for tc in tcs:
                tc = dict(tc)  # shallow copy so we don't mutate caller's data
                rel = tc["image"]
                abs_src = (
                    rel
                    if os.path.isabs(rel)
                    else os.path.join(self.save_dir, rel)
                )
                # Desired destination
                fname = os.path.basename(abs_src)
                dst = os.path.join(self._images_dir, fname)
                if not os.path.abspath(abs_src) == os.path.abspath(dst):
                    os.makedirs(self._images_dir, exist_ok=True)
                    shutil.copy2(abs_src, dst)
                tc["image"] = os.path.join("images", fname)
                updated[bid].append(tc)
        return updated
