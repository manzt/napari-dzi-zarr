import dask.array as da
import zarr
from napari_plugin_engine import napari_hook_implementation

from pathlib import Path
from .store import DZIStore


@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, list):
        return None

    if not path.endswith(".dzi"):
        return None

    return reader_function


def init_zarr_store(path):
    return DZIStore(path)


def reader_function(path):
    store = init_zarr_store(path)
    grp = zarr.open(store, mode="r")
    datasets = grp.attrs["multiscales"][0]["datasets"]
    data = [da.from_zarr(store, component=d["path"]) for d in datasets]
    add_kwargs = {"name": Path(path).name}
    return [(data, add_kwargs)]
