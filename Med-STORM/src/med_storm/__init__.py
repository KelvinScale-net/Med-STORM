# Expose package version metadata for external consumers.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("med_storm")
except PackageNotFoundError:  # pragma: no cover – during local development
    __version__ = "0.0.0"
