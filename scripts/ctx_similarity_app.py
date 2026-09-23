#!/usr/bin/env python3
"""Local Streamlit app for CTX click-to-similar image exploration."""

from pathlib import Path
from typing import Optional
import argparse
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from ctx_explorer.embeddings import DINOv3Extractor, DINOv3HFExtractor
from ctx_explorer.patch_retrieval import CTXPatchIndex
from ctx_explorer.retrieval import CTXSimilarityIndex

Image.MAX_IMAGE_PIXELS = None


def _read_model_sidecar(index_dir: Path) -> dict:
    sidecar = index_dir / "faiss.model.json"
    if sidecar.exists():
        with open(sidecar) as fp:
            return json.load(fp)
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--index-dir", type=Path, default=Path("outputs/ctx_similarity"))
    return parser.parse_args()


@st.cache_resource
def load_index(index_dir: Path):
    index = CTXSimilarityIndex(
        index_path=index_dir / "faiss.index",
        metadata_path=index_dir / "metadata.parquet",
    )
    return index


@st.cache_resource
def load_patch_index(index_dir: Path):
    patch_faiss = index_dir / "patches.faiss"
    if not patch_faiss.exists():
        return None
    return CTXPatchIndex(index_dir)


@st.cache_resource
def load_extractor(
    model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
    device: str = "cuda",
    image_size: int = 512,
):
    if "/" in model_name:
        extractor = DINOv3HFExtractor(
            model_name=model_name, device=device, use_half_precision=False
        )
        transform = DINOv3HFExtractor.get_default_transforms(image_size=image_size)
    else:
        extractor = DINOv3Extractor(
            model_name=model_name, device=device, use_half_precision=False
        )
        transform = DINOv3Extractor.get_default_transforms(image_size=image_size)
    return extractor, transform


def embed_crop(extractor, transform, crop_image: Image.Image) -> np.ndarray:
    """Run the DINO extractor on a single PIL crop and return its embedding vector."""
    if crop_image.mode != "RGB":
        crop_image = crop_image.convert("RGB")
    tensor = transform(crop_image).unsqueeze(0)
    embedding = extractor.extract(tensor)
    return embedding[0]


def snap_scale(crop_px: int, indexed_scales: list[int]) -> int:
    """Pick the indexed scale whose tile size is closest (in log-space) to the crop."""
    if not indexed_scales:
        return 0
    log_crop = np.log(max(crop_px, 1))
    return int(min(indexed_scales, key=lambda s: abs(np.log(s) - log_crop)))


def highlight_matching_window(
    tile_path: str,
    match_map: list[float],
    num_patches_per_tile: int,
    tile_scale_px: int,
    query_size_px: int,
) -> Image.Image:
    """Draw a bbox on the full tile around the best sub-window of size
    `query_size_px` (in the tile's pixel units).

    Method: slide a square window of size N patches across the tile's patch
    grid, where N = round(grid_n * query_size_px / tile_scale_px). For each
    position, score = sum of match_map values inside. Pick the max-scoring
    position and draw a green rectangle at its pixel coordinates on the full
    returned tile.
    """
    from PIL import ImageDraw

    grid_n = int(round(np.sqrt(num_patches_per_tile)))
    if grid_n * grid_n != num_patches_per_tile:
        grid_n = int(np.floor(np.sqrt(num_patches_per_tile)))

    tile = Image.open(tile_path).convert("RGB")
    w, h = tile.size
    mm = np.asarray(match_map, dtype=np.float32)
    if mm.size != grid_n * grid_n:
        return tile
    mm = mm.reshape(grid_n, grid_n)

    if tile_scale_px <= 0:
        tile_scale_px = max(w, h)
    ratio = max(1 / grid_n, min(1.0, query_size_px / float(tile_scale_px)))
    win_n = max(1, int(round(grid_n * ratio)))
    win_n = min(win_n, grid_n)

    if win_n == grid_n:
        best_r, best_c = 0, 0
    else:
        # Integral image for fast sum-over-window
        integral = np.zeros((grid_n + 1, grid_n + 1), dtype=np.float32)
        integral[1:, 1:] = np.cumsum(np.cumsum(mm, axis=0), axis=1)
        positions = grid_n - win_n + 1
        best_sum = -np.inf
        best_r = best_c = 0
        for r in range(positions):
            for c in range(positions):
                s = (
                    integral[r + win_n, c + win_n]
                    - integral[r, c + win_n]
                    - integral[r + win_n, c]
                    + integral[r, c]
                )
                if s > best_sum:
                    best_sum = float(s)
                    best_r, best_c = r, c

    cell_w = w / grid_n
    cell_h = h / grid_n
    x0 = int(best_c * cell_w)
    y0 = int(best_r * cell_h)
    x1 = int((best_c + win_n) * cell_w)
    y1 = int((best_r + win_n) * cell_h)

    draw = ImageDraw.Draw(tile)
    line_w = max(3, int(min(w, h) * 0.012))
    draw.rectangle([x0, y0, x1, y1], outline="#00ffae", width=line_w)
    return tile


def crop_to_match_region(
    tile_path: str,
    match_map: list[float],
    num_patches_per_tile: int,
    bbox_threshold: float = 0.6,
    margin_frac: float = 0.15,
) -> Image.Image:
    """Crop the tile to the bounding box of its high-matching patches.

    Returns a sub-image of `tile_path` covering the region where the query's
    features matched. `bbox_threshold` is an absolute cosine similarity floor
    (patches at least this similar to some query patch are considered part of
    the match). `margin_frac` expands the bbox by that fraction of the tile
    edge on each side so you see a little context around the match.

    If nothing in the tile clears the threshold, returns the whole tile (it
    was still ranked into top-k by aggregate sim, even if no single patch was
    strongly similar).
    """
    grid_n = int(round(np.sqrt(num_patches_per_tile)))
    if grid_n * grid_n != num_patches_per_tile:
        grid_n = int(np.floor(np.sqrt(num_patches_per_tile)))

    tile = Image.open(tile_path).convert("RGB")
    w, h = tile.size
    mm = np.asarray(match_map, dtype=np.float32)
    if mm.size != grid_n * grid_n:
        return tile
    mm = mm.reshape(grid_n, grid_n)

    mask = mm >= bbox_threshold
    if not mask.any():
        return tile

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    cell_w = w / grid_n
    cell_h = h / grid_n
    margin_px_x = int(margin_frac * w)
    margin_px_y = int(margin_frac * h)
    x0 = max(0, int(c0 * cell_w) - margin_px_x)
    y0 = max(0, int(r0 * cell_h) - margin_px_y)
    x1 = min(w, int((c1 + 1) * cell_w) + margin_px_x)
    y1 = min(h, int((r1 + 1) * cell_h) + margin_px_y)
    return tile.crop((x0, y0, x1, y1))


def highlight_match_map(
    tile_path: str,
    match_map: list[float],
    num_patches_per_tile: int,
    heat_low: float = 0.5,
    heat_high: float = 0.9,
    bbox_threshold: float = 0.75,
) -> Image.Image:
    """Overlay a heatmap of per-patch match scores on the tile using ABSOLUTE
    cosine similarities.

    match_map[i] is the max cosine similarity (in [-1, 1], typically [0, 1] for
    semantic matches) between the query's patches and the i-th indexed patch of
    this tile. We map that to a green overlay:

      - sim <= heat_low      → no overlay (the patch is clearly unrelated)
      - heat_low < sim < heat_high → linear alpha
      - sim >= heat_high     → full alpha

    A bbox is drawn around patches whose sim exceeds `bbox_threshold` (absolute).
    """
    from PIL import ImageDraw

    grid_n = int(round(np.sqrt(num_patches_per_tile)))
    if grid_n * grid_n != num_patches_per_tile:
        grid_n = int(np.floor(np.sqrt(num_patches_per_tile)))

    tile = Image.open(tile_path).convert("RGBA")
    w, h = tile.size
    cell_w = w / grid_n
    cell_h = h / grid_n

    mm = np.asarray(match_map, dtype=np.float32)
    if mm.size != grid_n * grid_n:
        return tile.convert("RGB")
    mm = mm.reshape(grid_n, grid_n)

    if heat_high <= heat_low:
        heat_high = heat_low + 1e-6
    intensity = np.clip((mm - heat_low) / (heat_high - heat_low), 0.0, 1.0)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for row in range(grid_n):
        for col in range(grid_n):
            v = float(intensity[row, col])
            if v <= 0.0:
                continue
            alpha = int(180 * v)
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w)
            y1 = int((row + 1) * cell_h)
            overlay_draw.rectangle([x0, y0, x1, y1], fill=(0, 255, 174, alpha))
    tile_with_heatmap = Image.alpha_composite(tile, overlay)

    mask = mm >= bbox_threshold
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        draw = ImageDraw.Draw(tile_with_heatmap)
        x0 = int(c0 * cell_w)
        y0 = int(r0 * cell_h)
        x1 = int((c1 + 1) * cell_w)
        y1 = int((r1 + 1) * cell_h)
        line_w = max(3, int(min(w, h) * 0.01))
        draw.rectangle([x0, y0, x1, y1], outline="#00ffae", width=line_w)

    return tile_with_heatmap.convert("RGB")


def _caption_for(path_str: str) -> str:
    path = Path(path_str)
    return f"{path.parent.name}/{path.name}"


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="CTX Similarity Explorer", layout="wide")
    st.title("CTX Similarity Explorer")

    st.sidebar.header("Settings")
    index_dir = st.sidebar.text_input("Index directory", value=str(args.index_dir))
    top_k = st.sidebar.slider("Neighbors", min_value=3, max_value=30, value=12)
    page_size = st.sidebar.slider(
        "Gallery page size", min_value=12, max_value=120, value=36, step=12
    )

    index = load_index(Path(index_dir))
    metadata = index.metadata.copy()
    metadata["image_path"] = metadata["image_path"].astype(str)

    st.sidebar.write(f"Indexed images: {len(metadata):,}")

    search_text = st.sidebar.text_input("Filter image path", value="")
    if search_text.strip():
        metadata = metadata[metadata["image_path"].str.contains(search_text.strip(), case=False)]

    if "sol" in metadata.columns:
        sol_values = sorted([int(value) for value in metadata["sol"].dropna().unique()])
        if len(sol_values) > 0:
            min_sol = int(sol_values[0])
            max_sol = int(sol_values[-1])
            selected_sol_range = st.sidebar.slider(
                "Sol range",
                min_value=min_sol,
                max_value=max_sol,
                value=(min_sol, max_sol),
            )
            metadata = metadata[
                metadata["sol"].isna()
                | (
                    (metadata["sol"] >= selected_sol_range[0])
                    & (metadata["sol"] <= selected_sol_range[1])
                )
            ]

    if "parent_dir" in metadata.columns:
        folder_values = sorted([str(value) for value in metadata["parent_dir"].dropna().unique()])
        if len(folder_values) > 1:
            selected_folders = st.sidebar.multiselect(
                "Folders",
                options=folder_values,
                default=folder_values,
            )
            metadata = metadata[metadata["parent_dir"].isin(selected_folders)]

    has_geo = "center_lon" in metadata.columns and "center_lat" in metadata.columns
    if has_geo:
        geo_rows = metadata[metadata["center_lon"].notna() & metadata["center_lat"].notna()]
        if len(geo_rows) > 0:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Region Filter")

            lon_min = float(geo_rows["center_lon"].min())
            lon_max = float(geo_rows["center_lon"].max())
            lat_min = float(geo_rows["center_lat"].min())
            lat_max = float(geo_rows["center_lat"].max())

            selected_lon = st.sidebar.slider(
                "Longitude",
                min_value=lon_min,
                max_value=lon_max,
                value=(lon_min, lon_max),
            )
            selected_lat = st.sidebar.slider(
                "Latitude",
                min_value=lat_min,
                max_value=lat_max,
                value=(lat_min, lat_max),
            )

            metadata = metadata[
                (
                    metadata["center_lon"].isna()
                    | (
                        (metadata["center_lon"] >= selected_lon[0])
                        & (metadata["center_lon"] <= selected_lon[1])
                    )
                )
                & (
                    metadata["center_lat"].isna()
                    | (
                        (metadata["center_lat"] >= selected_lat[0])
                        & (metadata["center_lat"] <= selected_lat[1])
                    )
                )
            ]

    if len(metadata) == 0:
        st.warning("No images match current filter.")
        return

    max_page = max(1, (len(metadata) - 1) // page_size + 1)
    page = st.sidebar.number_input("Page", min_value=1, max_value=max_page, value=1)

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(metadata))
    page_df = metadata.iloc[start_idx:end_idx].reset_index(drop=True)

    st.subheader("Gallery")
    st.caption("Click Select on any image to find similar images.")

    if "selected_row_id" not in st.session_state:
        st.session_state.selected_row_id = int(page_df.iloc[0]["row_id"])

    cols = st.columns(4)
    for idx, row in page_df.iterrows():
        col = cols[idx % 4]
        image_path = row["image_path"]
        row_id = int(row["row_id"])

        with col:
            st.image(image_path, caption=_caption_for(image_path), use_container_width=True)
            if st.button("Select", key=f"select_{row_id}"):
                st.session_state.selected_row_id = row_id

    st.divider()
    selected_row_id = int(st.session_state.selected_row_id)
    selected_row = index.metadata[index.metadata["row_id"] == selected_row_id].iloc[0]
    selected_path = str(selected_row["image_path"])

    st.subheader("Selected Image — drag the box to pick a region")
    query_mode = st.radio(
        "Query mode",
        options=["Whole tile", "Region crop"],
        horizontal=True,
    )

    selected_pil = Image.open(selected_path)

    index_dir_path = Path(index_dir)
    sidecar = _read_model_sidecar(index_dir_path)
    app_model_name = sidecar.get("model_name") or "dinov2_vitb14"
    app_image_size = sidecar.get("image_size") or 518

    if query_mode == "Region crop":
        # Optionally crop on the full source image instead of the small tile PNG.
        canvas_source = "Selected tile"
        source_path: Optional[Path] = None
        product_id = selected_row.get("product_id")
        if product_id is not None and isinstance(product_id, str):
            candidate_source = Path("data/raw/ctx") / f"{product_id}.tif"
            if candidate_source.exists():
                source_path = candidate_source
                canvas_source = st.radio(
                    "Crop canvas",
                    options=["Selected tile", "Full source image"],
                    horizontal=True,
                    key=f"canvas_source_{selected_row_id}",
                )

        if canvas_source == "Full source image" and source_path is not None:
            # Load source and present a downsampled preview in the cropper;
            # get the crop BOX and then re-crop from the full-res source so the
            # query uses native-resolution pixels.
            source_pil = Image.open(source_path)
            if source_pil.mode != "RGB":
                source_pil_rgb = source_pil.convert("RGB")
            else:
                source_pil_rgb = source_pil
            orig_w, orig_h = source_pil_rgb.size
            preview_max_edge = 1600
            scale_factor = min(1.0, preview_max_edge / max(orig_w, orig_h))
            if scale_factor < 1.0:
                preview_w = int(orig_w * scale_factor)
                preview_h = int(orig_h * scale_factor)
                preview = source_pil_rgb.resize(
                    (preview_w, preview_h), Image.LANCZOS
                )
            else:
                preview = source_pil_rgb

            st.caption(
                f"Source: {source_path.name} ({orig_w}×{orig_h} px, "
                f"preview downsampled by {1/scale_factor:.1f}× for drawing)"
            )

            box = st_cropper(
                preview,
                realtime_update=True,
                box_color="#00ffae",
                aspect_ratio=None,
                return_type="box",
                key=f"cropper_src_{selected_row_id}",
            )

            # Upscale box coords from preview → original coords
            inv = 1.0 / scale_factor if scale_factor > 0 else 1.0
            crop_left = int(box["left"] * inv)
            crop_top = int(box["top"] * inv)
            crop_right = int((box["left"] + box["width"]) * inv)
            crop_bottom = int((box["top"] + box["height"]) * inv)
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(orig_w, crop_right)
            crop_bottom = min(orig_h, crop_bottom)

            if crop_right <= crop_left or crop_bottom <= crop_top:
                st.warning("Draw a crop box on the source image.")
                return

            crop_pil = source_pil_rgb.crop(
                (crop_left, crop_top, crop_right, crop_bottom)
            )
        else:
            crop_pil = st_cropper(
                selected_pil,
                realtime_update=True,
                box_color="#00ffae",
                aspect_ratio=None,
                return_type="image",
                key=f"cropper_{selected_row_id}",
            )
        crop_w, crop_h = crop_pil.size
        if max(crop_w, crop_h) < 128:
            st.warning(
                f"Crop is very small ({crop_w}×{crop_h} px). It will be "
                f"upscaled to the model's 224 px input so DINO patches will "
                f"describe near-uniform content — retrieval will be noisy. "
                f"Draw a bigger box (≥128 px) for better matches."
            )
        indexed_scales = []
        if "tile_scale" in index.metadata.columns:
            indexed_scales = sorted(
                int(s) for s in index.metadata["tile_scale"].dropna().unique()
            )

        selected_scale = None
        if indexed_scales:
            crop_long_edge = max(crop_w, crop_h)
            selected_scale = snap_scale(crop_long_edge, indexed_scales)
            st.caption(
                f"Crop size: {crop_w}x{crop_h} px → querying scale {selected_scale}px "
                f"(available: {indexed_scales})"
            )
        else:
            st.caption(f"Crop size: {crop_w}x{crop_h} px (single-scale index)")

        try:
            extractor, transform = load_extractor(
                model_name=app_model_name, image_size=app_image_size
            )
        except Exception as exc:
            st.error(f"Failed to load embedding model ({app_model_name}): {exc}")
            return

        patch_index = load_patch_index(index_dir_path)
        if patch_index is not None:
            if crop_pil.mode != "RGB":
                crop_pil_rgb = crop_pil.convert("RGB")
            else:
                crop_pil_rgb = crop_pil
            import torch as _torch

            patch_image_size = int(
                patch_index.sidecar.get("image_size") or 224
            )
            patch_transform = DINOv3HFExtractor.get_default_transforms(
                image_size=patch_image_size
            )
            query_tensor = patch_transform(crop_pil_rgb).unsqueeze(0)
            if isinstance(extractor, DINOv3HFExtractor):
                query_patches = extractor.extract_patches(query_tensor)
            else:
                st.error(
                    "Patch index present but current extractor does not support "
                    "extract_patches(). Rebuild index with DINOv3HFExtractor."
                )
                return
            st.caption(
                f"Patch-level retrieval: {query_patches.shape[1]} query patches "
                f"vs {patch_index.index.ntotal:,} indexed patches"
            )
            tile_results = patch_index.query_by_patches(
                query_patches[0], k=top_k, tile_scale=selected_scale
            )
            # Reshape to look like CLS results for downstream rendering
            results = tile_results.rename(columns={"aggregate_sim": "similarity"})
            if "image_path" not in results.columns:
                st.warning("Patch index returned no matches.")
                return
        else:
            query_vector = embed_crop(extractor, transform, crop_pil)
            results = index.query_by_vector(
                query_vector, k=top_k, tile_scale=selected_scale
            )
    else:
        st.image(selected_path, caption=selected_path, use_container_width=True)
        results = index.query_by_row_id(selected_row_id, k=top_k, include_self=False)

    score_col = "similarity" if "similarity" in results.columns else "distance"

    st.subheader("Most Similar Images")
    st.caption("Higher similarity is better (cosine mode).")
    show_mode = "Full tile + matching window"
    if query_mode == "Region crop":
        show_mode = st.radio(
            "Result display",
            options=[
                "Full tile + matching window",
                "Crop to match",
                "Full tile + heatmap",
            ],
            horizontal=True,
            index=0,
        )

    patch_index_local = load_patch_index(index_dir_path) if query_mode == "Region crop" else None
    num_patches_per_tile = (
        int(patch_index_local.sidecar.get("num_patches_per_tile", 196))
        if patch_index_local is not None
        else 196
    )

    neighbor_cols = st.columns(4)
    for idx, row in results.iterrows():
        col = neighbor_cols[idx % 4]
        image_path = str(row["image_path"])
        score_val = float(row[score_col])
        caption_parts = [_caption_for(image_path), f"{score_col}: {score_val:.4f}"]
        if "coverage" in row and not pd.isna(row["coverage"]):
            caption_parts.append(f"cov: {int(row['coverage'])}/{num_patches_per_tile}")
        caption = "\n".join(caption_parts)

        with col:
            if patch_index_local is not None and "match_map" in row and row["match_map"] is not None:
                tile_scale_for_row = int(row.get("tile_scale") or 0)
                query_size_for_row = int(max(crop_w, crop_h)) if query_mode == "Region crop" else 0
                if show_mode == "Full tile + matching window":
                    image_to_show = highlight_matching_window(
                        image_path,
                        list(row["match_map"]),
                        num_patches_per_tile,
                        tile_scale_for_row,
                        query_size_for_row,
                    )
                elif show_mode == "Crop to match":
                    image_to_show = crop_to_match_region(
                        image_path,
                        list(row["match_map"]),
                        num_patches_per_tile,
                    )
                else:
                    image_to_show = highlight_match_map(
                        image_path,
                        list(row["match_map"]),
                        num_patches_per_tile,
                    )
                st.image(image_to_show, caption=caption, use_container_width=True)
            else:
                st.image(image_path, caption=caption, use_container_width=True)

    st.divider()
    st.subheader("Neighbor Table")
    display_cols = ["row_id", "image_path", score_col]
    extras = [
        c
        for c in [
            "filename",
            "parent_dir",
            "product_id",
            "sol",
            "center_lon",
            "center_lat",
            "source_image",
        ]
        if c in results.columns
    ]
    st.dataframe(results[display_cols + extras], use_container_width=True)


if __name__ == "__main__":
    main()
