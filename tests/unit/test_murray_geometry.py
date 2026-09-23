import numpy as np
from PIL import Image, ImageDraw

from ctx_explorer.embeddings.dinov3_hf import DINOv3HFExtractor
from scripts import murray_viewer, stream_murray_index


def test_bbox_ending_on_tile_boundary_covers_one_tile() -> None:
    z = 1
    tile_deg = murray_viewer.TILE_PX * murray_viewer.pixel_size_deg(z)
    lon_min = -180.0
    lon_max = lon_min + tile_deg
    lat_max = 90.0
    lat_min = lat_max - tile_deg

    assert murray_viewer._tile_range_for_bbox(
        z, lat_min, lat_max, lon_min, lon_max
    ) == (0, 0, 0, 0)
    assert list(
        stream_murray_index.enumerate_tiles(z, (lon_min, lat_min, lon_max, lat_max))
    ) == [(z, 0, 0)]


def test_bbox_crossing_tile_boundary_covers_both_tiles() -> None:
    z = 2
    tile_deg = murray_viewer.TILE_PX * murray_viewer.pixel_size_deg(z)
    lon_min = -180.0 + 0.75 * tile_deg
    lon_max = -180.0 + 1.25 * tile_deg
    lat_max = 90.0
    lat_min = lat_max - 0.5 * tile_deg

    assert murray_viewer._tile_range_for_bbox(
        z, lat_min, lat_max, lon_min, lon_max
    ) == (0, 1, 0, 0)


def test_candidate_cluster_anchors_include_all_offsets() -> None:
    anchors = set(murray_viewer._candidate_cluster_anchors(6, 10, 20, k_x=3, k_y=2))

    assert anchors == {
        (6, 8, 19),
        (6, 9, 19),
        (6, 10, 19),
        (6, 8, 20),
        (6, 9, 20),
        (6, 10, 20),
    }


def test_candidate_cluster_anchors_respect_world_edges() -> None:
    z = 3
    max_x = 2 * (2**z) - 1
    max_y = 1 * (2**z) - 1

    assert murray_viewer._candidate_cluster_anchors(z, 0, 0, k_x=3, k_y=3) == [
        (z, 0, 0)
    ]
    assert murray_viewer._candidate_cluster_anchors(
        z, max_x, max_y, k_x=3, k_y=3
    ) == [(z, max_x - 2, max_y - 2)]


def test_aspect_preserving_resize_pads_instead_of_restretching() -> None:
    img = Image.new("RGB", (100, 200), (0, 0, 0))
    ImageDraw.Draw(img).rectangle((0, 0, 99, 199), outline=(255, 255, 255), width=2)

    out = DINOv3HFExtractor.resize_preserve_aspect_and_pad(img, image_size=224)
    arr = np.asarray(out)
    ys, xs = np.where(arr[..., 0] > 20)

    assert out.size == (224, 224)
    assert ys.min() == 0
    assert ys.max() == 223
    assert 50 <= xs.min() <= 60
    assert 164 <= xs.max() <= 174
