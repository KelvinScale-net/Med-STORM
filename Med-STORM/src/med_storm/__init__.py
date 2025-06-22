import os

# Activate Pydantic v1 compatibility mode when running under Pydantic v2.
# This must be executed **before** any model classes are imported, therefore it
# lives at the very top-level package import.
os.environ.setdefault("PYDANTIC_V2_COMPAT_MODE", "1")

# Expose package version metadata for external consumers.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("med_storm")
except PackageNotFoundError:  # pragma: no cover – during local development
    __version__ = "0.0.0"
