try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# replace the asterisk with named imports
from .dzi_zarr import napari_get_reader, init_zarr_store
from .store import DZIStore


__all__ = ["napari_get_reader"]
