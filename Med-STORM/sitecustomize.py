"""Site customizations for Med-STORM test environments.

Ensures the project root directory is included in `sys.path` so that
imports like `import src.med_storm` work correctly regardless of the
current working directory (e.g., when running pytest from the `tests/`
folder).
"""

# ---------------------------------------------------------------------------
# Ensure Pydantic v1 compatibility when running under Pydantic v2
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("PYDANTIC_V2_COMPAT_MODE", "1")

import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure `src` namespace package import works even if the working directory changes
import types, importlib
if 'src' not in sys.modules:
    src_pkg = types.ModuleType('src')
    sys.modules['src'] = src_pkg
    # Attach actual package if present
    try:
        real_src = importlib.import_module('src')
        sys.modules['src'] = real_src
    except Exception:
        # Fallback: attempt to create namespace package mapping to project_root/src
        import importlib.machinery, importlib.util, pathlib as _pl
        src_path = _pl.Path(project_root) / 'src'
        if src_path.is_dir():
            spec = importlib.machinery.PathFinder().find_spec('src', [str(project_root)])
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                sys.modules['src'] = module 