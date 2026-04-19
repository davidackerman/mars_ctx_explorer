#!/usr/bin/env python3
"""Local Streamlit app for CTX click-to-similar image exploration."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from scientific_pipelines.core.embeddings import DINOv3Extractor
from scientific_pipelines.planetary.mars.ctx.retrieval import CTXSimilarityIndex

Image.MAX_IMAGE_PIXELS = None


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
def load_extractor(model_name: str = "dinov2_vitb14", device: str = "cuda"):
    extractor = DINOv3Extractor(model_name=model_name, device=device, use_half_precision=False)
    transform = DINOv3Extractor.get_default_transforms()
    return extractor, transform


def embed_crop(extractor, transform, crop_image: Image.Image) -> np.ndarray:
    """Run the DINO extractor on a single PIL crop and return its embedding vector."""
    if crop_image.mode != "RGB":
        crop_image = crop_image.convert("RGB")
    tensor = transform(crop_image).unsqueeze(0)
    embedding = extractor.extract(tensor)
    return embedding[0]


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

    if query_mode == "Region crop":
        crop_pil = st_cropper(
            selected_pil,
            realtime_update=True,
            box_color="#00ffae",
            aspect_ratio=None,
            return_type="image",
            key=f"cropper_{selected_row_id}",
        )
        st.caption(f"Crop size: {crop_pil.size[0]}x{crop_pil.size[1]} px")

        try:
            extractor, transform = load_extractor()
        except Exception as exc:
            st.error(f"Failed to load embedding model: {exc}")
            return

        query_vector = embed_crop(extractor, transform, crop_pil)
        results = index.query_by_vector(query_vector, k=top_k)
    else:
        st.image(selected_path, caption=selected_path, use_container_width=True)
        results = index.query_by_row_id(selected_row_id, k=top_k, include_self=False)

    score_col = "similarity" if "similarity" in results.columns else "distance"

    st.subheader("Most Similar Images")
    st.caption("Higher similarity is better (cosine mode).")

    neighbor_cols = st.columns(4)
    for idx, row in results.iterrows():
        col = neighbor_cols[idx % 4]
        image_path = str(row["image_path"])
        score_val = float(row[score_col])
        caption = f"{_caption_for(image_path)}\n{score_col}: {score_val:.4f}"

        with col:
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
