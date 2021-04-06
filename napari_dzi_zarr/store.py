import xml.etree.ElementTree as ET
from dataclasses import dataclass

import fsspec
import imageio
import numpy as np
from zarr.storage import init_array, init_group
from zarr.util import json_dumps


@dataclass
class DZIMetadata:
    tilesize: int
    overlap: int
    format: str
    height: int
    width: int

    @classmethod
    def from_xml(cls, text: str):
        Image = ET.fromstring(text)
        Size = Image[0]
        return cls(
            tilesize=int(Image.get("TileSize")),
            overlap=int(Image.get("Overlap")),
            format=Image.get("Format"),
            width=int(Size.get("Width")),
            height=int(Size.get("Height")),
        )


def _init_meta_store(dzi_meta: DZIMetadata, csize: int) -> dict:
    """Generates zarr metadata key-value mapping for all levels of DZI pyramid"""
    store = dict()

    # DZI generates all levels of the pyramid
    # Level 0 is 1x1 image, so we need to calculate the max level (highest resolution)
    # and trim the pyramid to just the tiled levels.
    max_size = max(dzi_meta.width, dzi_meta.height)
    max_level = np.ceil(np.log2(max_size)).astype(int)

    nlevels = max_level - np.ceil(np.log2(dzi_meta.tilesize)).astype(int)
    levels = list(reversed(range(max_level + 1)))[:nlevels]

    # Create root group
    init_group(store)

    # Create root attrs (multiscale meta)
    datasets = [dict(path=str(i)) for i in levels]
    root_attrs = dict(multiscales=[dict(datasets=datasets, version="0.1")])
    store[".zattrs"] = json_dumps(root_attrs)

    # Create zarr array meta for each level of DZI pyramid
    for level in range(nlevels):
        init_array(
            store=store,
            path=f"{max_level - level}",
            shape=(dzi_meta.height // 2 ** level, dzi_meta.width // 2 ** level, csize),
            chunks=(dzi_meta.tilesize, dzi_meta.tilesize, csize),
            compressor=None,  # chunk is decoded with store, so no zarr compression
            dtype="|u1",  # RGB/A images only
        )

    return store


def _normalize_chunk(
    arr: np.ndarray, x: int, y: int, dzi_meta: DZIMetadata
) -> np.ndarray:
    """Transforms DZI tiles to uniformly sized zarr array chunks"""
    # https://github.com/openseadragon/openseadragon/wiki/The-DZI-File-Format#overlap
    # Here we trim overlapping tiles or pad edge tiles based on the chunk key.
    size_y, size_x, _ = arr.shape
    tilesize, overlap = dzi_meta.tilesize, dzi_meta.overlap

    if overlap == 0:
        # No overlap; no need to trim overlap.
        view = arr
    else:
        # Easy to detect top/left.
        top_edge = y == 0
        left_edge = x == 0

        # How much overlap to expect if not an edge.
        overlap_y = overlap if top_edge else 2 * overlap
        overlap_x = overlap if left_edge else 2 * overlap

        # If tile is not full size plus both overlaps then it must
        # be a left or bottom edge.
        bottom_edge = size_y < tilesize + overlap_y
        right_edge = size_x < tilesize + overlap_x

        # Trim overlaps based on whether we are interior/edge/corner.
        y0 = None if top_edge else overlap
        y1 = None if bottom_edge else -overlap
        x0 = None if left_edge else overlap
        x1 = None if right_edge else -overlap

        view = arr[y0:y1, x0:x1, :]

    if view.shape[0] < tilesize or view.shape[1] < tilesize:
        # Pad tiles out to tilesize if needed.
        y_pad = tilesize - view.shape[0]
        x_pad = tilesize - view.shape[1]
        view = np.pad(view, ((0, y_pad), (0, x_pad), (0, 0)))

    return view


class DZIStore:
    def __init__(self, url: str, *, pilmode="RGB", **storage_options):
        fs, meta_path = fsspec.core.url_to_fs(url, **storage_options)
        self.fs = fs
        self.root = meta_path.rsplit(".", 1)[0] + "_files"
        self._dzi_meta = DZIMetadata.from_xml(fs.cat(meta_path))

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
        except Exception:
            raise KeyError(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def keys(self):
        return self._meta_store.keys()

    def __iter__(self):
        return iter(self._meta_store)
