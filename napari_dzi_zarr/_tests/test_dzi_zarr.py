from napari_dzi_zarr import napari_get_reader


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None


def test_get_dzi():
    reader = napari_get_reader("my_file.dzi")
    return reader is not None
