"""Attack method registry for test case generation."""

import importlib


def load_method(method_name: str, **kwargs):
    """Load an attack method by name.

    Args:
        method_name: Name of the method module under pipeline/methods/
        **kwargs: Passed to the method's constructor

    Returns:
        An instance of the method's AttackMethod class
    """
    module = importlib.import_module(f"methods.{method_name}")
    return module.AttackMethod(**kwargs)
