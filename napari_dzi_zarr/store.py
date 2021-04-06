import itertools
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import fsspec
import imageio
import numpy as np
from zarr.storage import init_array, init_group
from zarr.util import json_dumps, json_loads


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


def _normalize_chunk(
    arr: np.ndarray, x: int, y: int, dzi_meta: DZIMetadata
) -> np.ndarray:
    """Transforms DZI tiles to uniformly sized zarr array chunks"""
    # https://github.com/openseadragon/openseadragon/wiki/The-DZI-File-Format#overlap
    # Here we trim overlapping tiles or pad edge tiles based on the chunk key.
    size_y, size_x = arr.shape[0], arr.shape[1]
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

        view = arr[y0:y1, x0:x1]

    if view.shape[0] < tilesize or view.shape[1] < tilesize:
        # Pad tiles out to tilesize if needed.
        y_pad = tilesize - view.shape[0]
        x_pad = tilesize - view.shape[1]
        pad_width = ((0, y_pad), (0, x_pad))
        view = np.pad(view, pad_width + ((0, 0),) if arr.ndim == 3 else pad_width)

    return view


class DZIStore:
    def __init__(self, url: str, **storage_options):
        fs, meta_path = fsspec.core.url_to_fs(url, **storage_options)
        self.fs = fs
        self.root = meta_path.rsplit(".", 1)[0] + "_files"
        self._dzi_meta = DZIMetadata.from_xml(fs.cat(meta_path))
        self._meta_store = self._init_meta_store()

    def _read_tile(self, level, x, y):
        path = f"{self.root}/{level}/{x}_{y}.{self._dzi_meta.format}"
        # Read bytes from abstract file system
        cbytes = self.fs.cat(path)
        # Decode bytes as image tile
        tile = imageio.imread(cbytes)
        return tile

    def _init_meta_store(self) -> dict:
        """Generates zarr metadata key-value mapping for all levels of DZI pyramid"""
        store = dict()

        # DZI generates all levels of the pyramid
        # Level 0 is 1x1 image, so we need to calculate the max level (highest resolution)
        # and trim the pyramid to just the tiled levels.
        max_size = max(self._dzi_meta.width, self._dzi_meta.height)
        max_level = np.ceil(np.log2(max_size)).astype(int)

        nlevels = max_level - np.ceil(np.log2(self._dzi_meta.tilesize)).astype(int)
        levels = list(reversed(range(max_level + 1)))[:nlevels]

        # Create root group
        init_group(store)

        # Create root attrs (multiscale meta)
        datasets = [dict(path=str(i)) for i in levels]
        root_attrs = dict(multiscales=[dict(datasets=datasets, version="0.1")])
        store[".zattrs"] = json_dumps(root_attrs)

        # Grab a representative tile image
        tile = self._read_tile(max_level, 0, 0)
        csize = None if tile.ndim == 2 else tile.shape[-1]

        # Create zarr array meta for each level of DZI pyramid
        for level in range(nlevels):
            shape = (
                self._dzi_meta.height // 2 ** level,
                self._dzi_meta.width // 2 ** level,
            )
            chunks = (self._dzi_meta.tilesize, self._dzi_meta.tilesize)
            init_array(
                store=store,
                path=f"{max_level - level}",
                shape=shape + (csize,) if csize else shape,
                chunks=chunks + (csize,) if csize else chunks,
                compressor=None,  # chunk is decoded with store, so no zarr compression
                dtype="|u1",  # RGB/A images only
            )
        return store

    def __getitem__(self, key):
        if key in self._meta_store:
            return self._meta_store[key]

        try:
            # Transform key to DZI path
            level, chunk_key = key.split("/")
            parts = chunk_key.split(".")
            x, y = parts[1], parts[0]
            tile = self._read_tile(level, x, y)
        except Exception:
            raise KeyError(key)

        # Normalize DZI tile as zarr chunk
        trimmed_tile = _normalize_chunk(
            arr=tile, x=int(x), y=int(y), dzi_meta=self._dzi_meta
        )
        return trimmed_tile.tobytes()

    def __setitem__(self, key, value):
        raise NotImplementedError

    def keys(self):
        return self._meta_store.keys()

    def __iter__(self):
        return iter(self._meta_store)

    def write_fsspec(self, filename: str = None):
        if self._dzi_meta.overlap != 0:
            raise ValueError(
                "Tile overlap must be 0 for DZI source to write reference."
            )

        spec = dict(version=1, templates=dict(u=self.root), refs={})
        compressors = dict(
            jpeg="imagecodecs_jpeg",
            jpg="imagecodecs_jpeg",
            png="imagecodecs_png",
        )
        codec_id = compressors[self._dzi_meta.format]

        for k, v in self._meta_store.items():
            if k.endswith(".zarray"):
                # Decode array metadata
                meta = json_loads(v)

                # Create generated entry for tile/chunk references
                shape = meta["shape"]
                chunks = meta["chunks"]

                for (i, j) in itertools.product(
                    # Cannot map variable sized chunks to zarr data model.
                    # We don't add edge tiles to the referend (right, and bottom)
                    range(math.floor(shape[1] / chunks[1])),
                    range(math.floor(shape[0] / chunks[0])),
                ):
                    arr_key = k.rstrip(".zarray/")
                    key = f"{arr_key}/{j}.{i}" + (".0" if len(shape) == 3 else "")
                    url = "{{u}}" + f"/{arr_key}/{i}_{j}.{self._dzi_meta.format}"
                    spec["refs"][key] = [url]

                # Override `null` compressor
                meta["compressor"] = dict(id=codec_id)
                v = json.dumps(meta, indent=1).encode("ascii")

            # Write `.zattrs`, `.zgroup`, and modified `.zarray` metadata
            spec["refs"][k] = v.decode()

        if filename is None:
            # return refs directly as a dict
            return spec

        with open(filename, mode="w") as fh:
            fh.write(json.dumps(spec, indent=1))
