"""Patch-level similarity retrieval for CTX tiles.

Complementary to CTXSimilarityIndex (which stores one CLS embedding per tile),
this module stores *all DINO patch embeddings* per tile. A region-crop query
extracts its own patch tokens and retrieves tiles whose patches best match the
query patches, enabling true "this region appears inside another tile" search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


Image.MAX_IMAGE_PIXELS = None


@dataclass
class PatchIndexPaths:
    index_dir: Path

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "patches.faiss"

    @property
    def patch_meta_path(self) -> Path:
        return self.index_dir / "patches_metadata.parquet"

    @property
    def tile_meta_path(self) -> Path:
        return self.index_dir / "patches_tiles.parquet"

    @property
    def sidecar_path(self) -> Path:
        return self.index_dir / "patches.model.json"


class CTXPatchIndex:
    """FAISS-backed index over per-tile patch embeddings.

    Metadata layout:
      patches_metadata.parquet: one row per patch. Columns: row_id (faiss id),
        tile_row_id (compact index into tile table), patch_idx (0..P-1).
      patches_tiles.parquet: one row per tile. Columns: tile_row_id, image_path,
        tile_scale, product_id, plus whatever columns the normal metadata has.
    """

    def __init__(self, index_dir: Path):
        import faiss

        self.paths = PatchIndexPaths(Path(index_dir))
        if not self.paths.faiss_path.exists():
            raise FileNotFoundError(f"Patch FAISS index not found: {self.paths.faiss_path}")

        self.index = faiss.read_index(str(self.paths.faiss_path))
        self.patch_meta = pd.read_parquet(self.paths.patch_meta_path)
        self.tiles = pd.read_parquet(self.paths.tile_meta_path)

        if self.paths.sidecar_path.exists():
            with open(self.paths.sidecar_path) as fp:
                self.sidecar = json.load(fp)
        else:
            self.sidecar = {}

    @classmethod
    def build_from_tiles(
        cls,
        tile_paths: List[Path],
        extractor,
        transform,
        index_dir: Path,
        batch_size: int = 32,
        device: str = "cuda",
        normalize: bool = True,
        model_name: Optional[str] = None,
        image_size: Optional[int] = None,
        extra_tile_metadata: Optional[pd.DataFrame] = None,
    ) -> "CTXPatchIndex":
        """Extract patch embeddings for every tile and persist a flat FAISS index.

        Uses a disk-backed memmap to avoid holding two full copies of a ~tens-of-GB
        patch matrix in RAM.

        Args:
            tile_paths: paths to tile PNGs
            extractor: DINOv3HFExtractor (needs extract_patches)
            transform: torchvision transform to 3-channel RGB tensor
            index_dir: output directory
            batch_size: tiles per forward pass
            normalize: L2-normalize patch vectors before indexing (cosine sim)
            extra_tile_metadata: optional DataFrame keyed by image_path to merge
                into the tile-level metadata parquet (e.g. tile_scale, product_id)
        """
        import faiss

        paths = PatchIndexPaths(Path(index_dir))
        paths.index_dir.mkdir(parents=True, exist_ok=True)

        # First pass: probe one tile to learn D, P.
        probe_img = Image.open(tile_paths[0]).convert("RGB")
        probe_tensor = transform(probe_img).unsqueeze(0)
        probe_patches = extractor.extract_patches(probe_tensor)
        num_patches_per_tile = int(probe_patches.shape[1])
        dim = int(probe_patches.shape[2])
        del probe_patches, probe_tensor, probe_img

        total_patches = len(tile_paths) * num_patches_per_tile
        memmap_path = paths.index_dir / "patches.memmap.f32"
        memmap_path.unlink(missing_ok=True)
        patches_matrix = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(total_patches, dim),
        )
        logger.info(
            f"Allocating memmap {memmap_path} "
            f"(shape=({total_patches:,}, {dim}), "
            f"{total_patches * dim * 4 / 1e9:.1f} GB)"
        )

        tile_rows = []
        batch_tensors: List[torch.Tensor] = []
        batch_tile_ids: List[int] = []

        def flush():
            if not batch_tensors:
                return
            imgs = torch.stack(batch_tensors, dim=0)
            patches = extractor.extract_patches(imgs)  # (B, P, D)
            for local_i, tile_id in enumerate(batch_tile_ids):
                start = tile_id * num_patches_per_tile
                end = start + num_patches_per_tile
                patches_matrix[start:end] = patches[local_i].astype(np.float32)
            batch_tensors.clear()
            batch_tile_ids.clear()

        for tile_row_id, tile_path in enumerate(
            tqdm(tile_paths, desc="Extracting patch tokens")
        ):
            try:
                img = Image.open(tile_path).convert("RGB")
                tensor = transform(img)
            except Exception as exc:
                logger.warning(f"Skipping {tile_path}: {exc}")
                continue

            batch_tensors.append(tensor)
            batch_tile_ids.append(tile_row_id)
            tile_rows.append(
                {
                    "tile_row_id": tile_row_id,
                    "image_path": str(tile_path),
                }
            )

            if len(batch_tensors) >= batch_size:
                flush()

        flush()
        patches_matrix.flush()
        logger.info(
            f"Patch matrix populated: shape={patches_matrix.shape}, "
            f"dim={dim}, patches_per_tile={num_patches_per_tile}"
        )

        # FAISS build: normalize + add in chunks to avoid peak memory spike
        if normalize:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

        chunk_rows = 1_000_000
        for start in tqdm(
            range(0, total_patches, chunk_rows), desc="FAISS add"
        ):
            end = min(start + chunk_rows, total_patches)
            chunk = np.ascontiguousarray(patches_matrix[start:end])
            if normalize:
                faiss.normalize_L2(chunk)
            index.add(chunk)
            del chunk

        faiss.write_index(index, str(paths.faiss_path))

        # Per-patch metadata: row_id, tile_row_id, patch_idx
        n_tiles = len(tile_rows)
        tile_ids_col = np.repeat(
            np.arange(n_tiles, dtype=np.int64), num_patches_per_tile
        )
        patch_ids_col = np.tile(
            np.arange(num_patches_per_tile, dtype=np.int32), n_tiles
        )
        patch_meta = pd.DataFrame(
            {
                "row_id": np.arange(len(tile_ids_col), dtype=np.int64),
                "tile_row_id": tile_ids_col,
                "patch_idx": patch_ids_col,
            }
        )
        patch_meta.to_parquet(paths.patch_meta_path, index=False)

        # Per-tile metadata
        tile_meta = pd.DataFrame(tile_rows)
        if extra_tile_metadata is not None and len(extra_tile_metadata) > 0:
            tile_meta = tile_meta.merge(
                extra_tile_metadata, on="image_path", how="left"
            )
        tile_meta.to_parquet(paths.tile_meta_path, index=False)

        sidecar = {
            "model_name": model_name,
            "image_size": image_size,
            "embedding_dim": int(dim),
            "num_patches_per_tile": int(num_patches_per_tile),
            "normalize": bool(normalize),
        }
        with open(paths.sidecar_path, "w") as fp:
            json.dump(sidecar, fp, indent=2)

        logger.info(
            f"Built patch index at {paths.faiss_path} with {index.ntotal} vectors "
            f"({n_tiles} tiles × {num_patches_per_tile} patches, dim={dim})"
        )
        return cls(paths.index_dir)

    def query_by_patches(
        self,
        query_patches: np.ndarray,
        k: int = 12,
        k_per_query_patch: int = 500,
        tile_scale: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve tiles by aggregating patch-level nearest neighbors.

        For each query patch, pull `k_per_query_patch` nearest indexed patches
        from FAISS. Each hit maps to a tile. For each candidate tile, compute
        a coverage-weighted score:

            score = sum(best-sim per query patch, missing = 0) / num_query_patches

        Missing query patches contribute 0, so tiles that cover only a small
        subset of the query are penalized vs tiles that cover it broadly.
        """
        import faiss

        if query_patches.ndim == 2:
            # Single-image shape (P, D) → (1, P, D)
            query_patches = query_patches[np.newaxis, ...]
        if query_patches.ndim != 3 or query_patches.shape[0] != 1:
            raise ValueError(
                "query_patches must be of shape (P, D) or (1, P, D)"
            )

        q = query_patches[0].astype(np.float32)  # (P, D)
        if self.sidecar.get("normalize", True):
            faiss.normalize_L2(q)

        distances, indices = self.index.search(q, k_per_query_patch)
        # distances, indices: (P, k_per_query_patch)

        num_query_patches = q.shape[0]
        tile_row_for_hit = self.patch_meta["tile_row_id"].to_numpy()
        patch_idx_for_hit = self.patch_meta["patch_idx"].to_numpy()

        # For each tile, keep best similarity per query patch, and remember
        # which indexed patch produced it.
        best_sim_per_qp: dict[int, np.ndarray] = {}
        best_patch_idx_per_qp: dict[int, np.ndarray] = {}

        for qp_idx in range(num_query_patches):
            for rank in range(indices.shape[1]):
                patch_row_id = int(indices[qp_idx, rank])
                if patch_row_id < 0:
                    continue
                sim = float(distances[qp_idx, rank])
                tile_id = int(tile_row_for_hit[patch_row_id])
                hit_patch_idx = int(patch_idx_for_hit[patch_row_id])

                if tile_id not in best_sim_per_qp:
                    best_sim_per_qp[tile_id] = np.zeros(
                        num_query_patches, dtype=np.float32
                    )
                    best_patch_idx_per_qp[tile_id] = np.full(
                        num_query_patches, -1, dtype=np.int32
                    )
                if sim > best_sim_per_qp[tile_id][qp_idx]:
                    best_sim_per_qp[tile_id][qp_idx] = sim
                    best_patch_idx_per_qp[tile_id][qp_idx] = hit_patch_idx

        if not best_sim_per_qp:
            return pd.DataFrame(
                columns=["tile_row_id", "aggregate_sim", "coverage", "best_patch_idx"]
            )

        scored = []
        for tile_id, best_vec in best_sim_per_qp.items():
            coverage = int((best_vec > 0).sum())
            # Coverage-weighted: missing query patches count as 0
            aggregate = float(best_vec.sum() / num_query_patches)
            # Also track which single indexed patch best matched any query patch
            best_patch = int(
                best_patch_idx_per_qp[tile_id][int(np.argmax(best_vec))]
            )
            scored.append(
                {
                    "tile_row_id": tile_id,
                    "aggregate_sim": aggregate,
                    "coverage": coverage,
                    "best_patch_idx": best_patch,
                }
            )

        score_df = pd.DataFrame(scored)
        tile_df = self.tiles.merge(score_df, on="tile_row_id", how="inner")
        tile_df = tile_df.sort_values("aggregate_sim", ascending=False)

        if tile_scale is not None and "tile_scale" in tile_df.columns:
            filtered = tile_df[tile_df["tile_scale"] == int(tile_scale)]
            if len(filtered) > 0:
                tile_df = filtered

        top = tile_df.head(k).reset_index(drop=True)

        # For each returned tile, compute the full per-patch match map:
        # for each of its 196 patches, the max cosine sim to any query patch.
        # Uses FAISS reconstruct_n to pull indexed patches directly.
        num_patches = int(self.sidecar.get("num_patches_per_tile") or 196)
        match_maps = []
        for tile_id in top["tile_row_id"].astype(int).tolist():
            start = int(tile_id) * num_patches
            tile_patches = self.index.reconstruct_n(start, num_patches)
            tile_patches = np.ascontiguousarray(tile_patches, dtype=np.float32)
            # Patches are already L2-normalized (normalize=True at build time).
            sim_matrix = q @ tile_patches.T  # (num_query_patches, num_patches)
            match_map = sim_matrix.max(axis=0)  # per tile-patch: best sim to any query patch
            match_maps.append(match_map.astype(np.float32).tolist())
        top["match_map"] = match_maps

        return top
