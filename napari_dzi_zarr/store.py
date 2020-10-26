import fsspec
from zarr.util import json_dumps
import numpy as np
import imageio

import xml.etree.ElementTree as ET
from collections import namedtuple

ZARR_FORMAT = 2
ZARR_META_KEY = ".zattrs"
ZARR_ARRAY_META_KEY = ".zarray"
ZARR_GROUP_META_KEY = ".zgroup"

DZIMetadata = namedtuple("DZIMetadata", "tilesize overlap format height width")


def _parse_DZI(xml_str: str) -> DZIMetadata:
    Image = ET.fromstring(xml_str)
    Size = Image[0]
    return DZIMetadata(
        tilesize=int(Image.get("TileSize")),
        overlap=int(Image.get("Overlap")),
        format=Image.get("Format"),
        width=int(Size.get("Width")),
        height=int(Size.get("Height")),
    )


def _init_meta_store(dzi_meta: DZIMetadata, csize: int) -> dict:
    """Generates zarr metadata key-value mapping for all levels of DZI pyramid"""
    d = dict()
    # DZI generates all levels of the pyramid
    # Level 0 is 1x1 image, so we need to calculate the max level (highest resolution)
    # and trim the pyramid to just the tiled levels.
    max_size = max(dzi_meta.width, dzi_meta.height)
    max_level = np.ceil(np.log2(max_size)).astype(int)

    nlevels = max_level - np.ceil(np.log2(dzi_meta.tilesize)).astype(int)
    levels = list(reversed(range(max_level + 1)))[:nlevels]

    # Create root group
    group_meta = dict(zarr_format=ZARR_FORMAT)
    d[ZARR_GROUP_META_KEY] = json_dumps(group_meta)

    # Create root attrs (multiscale meta)
    datasets = [dict(path=str(i)) for i in levels]
    root_attrs = dict(multiscales=[dict(datasets=datasets, version="0.1")])
    d[ZARR_META_KEY] = json_dumps(root_attrs)

    # Create zarr array meta for each level of DZI pyramid
    for level in range(nlevels):
        xsize, ysize = (dzi_meta.width // 2 ** level, dzi_meta.height // 2 ** level)
        arr_meta_key = f"{max_level - level}/{ZARR_ARRAY_META_KEY}"
        arr_meta = dict(
            shape=(ysize, xsize, csize),
            chunks=(dzi_meta.tilesize, dzi_meta.tilesize, csize),
            compressor=None,  # chunk is decoded with store, so no zarr compression
            dtype="|u1",  # RGB/A images only
            fill_value=0,
            filters=None,
            order="C",
            zarr_format=ZARR_FORMAT,
        )
        d[arr_meta_key] = json_dumps(arr_meta)

    return d


def _normalize_chunk(
    arr: np.ndarray, x: int, y: int, dzi_meta: DZIMetadata
) -> np.ndarray:
    """Transforms DZI tiles to uniformly sized zarr array chunks"""
    # https://github.com/openseadragon/openseadragon/wiki/The-DZI-File-Format#overlap
    # Here we trim overlapping tiles or pad edge tiles based on the chunk key.
    ysize, xsize, _ = arr.shape
    tilesize, overlap = dzi_meta.tilesize, dzi_meta.overlap

    if xsize == tilesize and ysize == tilesize:
        # Decoded image is already correct size.
        return arr

    # TODO: There is probably a more elegant way to do this...
    view = arr
    if xsize - tilesize == 2 * overlap:
        # Inner x; overlap on left and right
        view = view[:, overlap:-overlap, :]

    if ysize - tilesize == 2 * overlap:
        # Inner y; overlap on top and bottom
        view = view[overlap:-overlap, :, :]

    if xsize - tilesize == overlap:
        # Edge x; overlap on left or right
        xslice = slice(None, -overlap) if x == 0 else slice(overlap, None)
        view = view[:, xslice, :]

    if ysize - tilesize == overlap:
        # Edge y; overlap on top or bottom
        yslice = slice(None, -overlap) if y == 0 else slice(overlap, None)
        view = view[yslice, :, :]

    if view.shape[0] < tilesize or view.shape[1] < tilesize:
        # Tile is smaller than tilesize; Needs to be padded.
        y_pad = tilesize - view.shape[0]
        x_pad = tilesize - view.shape[1]
        return np.pad(view, ((0, y_pad), (0, x_pad), (0, 0)))

    return view


class DZIStore:
    def __init__(self, url: str, *, pilmode="RGB", **storage_options):
        fs, meta_path = fsspec.core.url_to_fs(url, **storage_options)
        self.fs = fs
        self.root = meta_path.rsplit(".", 1)[0] + "_files"
        self._dzi_meta = _parse_DZI(fs.cat(meta_path))

        if pilmode not in ["RGB", "RGBA"]:
            raise ValueError(f"pilmode must be 'RGB' or 'RGBA', got: {pilmode}")

        self.pilmode = pilmode
        self._meta_store = _init_meta_store(dzi_meta=self._dzi_meta, csize=len(pilmode))

    def __getitem__(self, key):
        if key in self._meta_store:
            return self._meta_store[key]

        try:
            # Transform key to DZI path
            level, chunk_key = key.split("/")
            y, x, _ = chunk_key.split(".")
            path = f"{self.root}/{level}/{x}_{y}.{self._dzi_meta.format}"
            # Read bytes from abstract file system
            cbytes = self.fs.cat(path)
            # Decode bytes as image tile
            tile = imageio.imread(cbytes, pilmode=self.pilmode)
            # Normalize DZI tile as zarr chunk
            trimmed_tile = _normalize_chunk(
                arr=tile, x=int(x), y=int(y), dzi_meta=self._dzi_meta
            )
            return trimmed_tile.tobytes()
        except:
            raise KeyError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def keys(self):
        return self._meta_store.keys()

    def __iter__(self):
        return iter(self._meta_store)
