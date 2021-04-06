try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# replace the asterisk with named imports
from .dzi_zarr import init_zarr_store, napari_get_reader
from .store import DZIStore

__all__ = ["napari_get_reader"]
