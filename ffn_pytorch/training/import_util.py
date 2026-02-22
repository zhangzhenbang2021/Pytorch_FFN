"""Functions for dynamically importing symbols from modules."""

import importlib


def import_symbol(specifier: str,
                  default_packages: str = 'ffn_pytorch.training.models'):
    """Imports a symbol from a module.

    Args:
        specifier: fully qualified name or just module.class format
        default_packages: default package prefix if not fully qualified

    Returns:
        The imported symbol
    """
    parts = specifier.rsplit('.', 1)
    if len(parts) == 1:
        raise ValueError(
            f'Specifier must have at least module.class format: {specifier}')

    module_path, class_name = parts

    for prefix in [module_path, f'{default_packages}.{module_path}', '']:
        try:
            if prefix:
                mod = importlib.import_module(prefix)
            else:
                mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue

    raise ImportError(f'Could not import {specifier}')
