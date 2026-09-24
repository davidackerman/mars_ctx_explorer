# Mars CTX Explorer

**Draw a box anywhere on Mars and find the places that look like it.**

The Murray Lab global CTX mosaic is cut into tiles, every tile is embedded with DINOv3, and the
embeddings live in a FAISS index. A Leaflet map sits on top: click or drag-select a region and
the backend embeds that patch, queries the index, and returns the most similar places on the
planet, with thumbnails and a cluster overlay so you can see where the matches concentrate.
Sub-tile queries use a patch-level index with GPU sliding-window matching, so a small feature
inside a tile can be searched for directly.

Once you can pull up "more like this" for any spot, the same index supports the original
question the project started with: **what kinds of terrain are here, and what is unusual?**
That is the terrain clustering pipeline below.

**Author**: David Ackerman

Split out of the `mars_astrobio` monorepo in September 2026. Siblings:
[mars_astrobio](https://github.com/davidackerman/mars_astrobio) (the dormant WATSON rover
biosignature project this grew out of) and
[backyard_worlds](https://github.com/davidackerman/backyard_worlds) (brown dwarf detection).

## Quick start: global Murray Lab index

```bash
pixi install
pixi shell

# Build a global CLS index at zoom 8 (streams tiles from the public mosaic, no local storage)
pixi run murray-index --zoom 8 --output-dir outputs/murray_z8_cls

# Optional: a patch-level index for sub-tile queries
pixi run murray-patch-index --zoom 8 --output-dir outputs/murray_z8_patch

# Serve the viewer, then open http://localhost:8000
pixi run murray-viewer --index-dir outputs/murray_z8_cls
```

`scripts/run_aspectpad_rebuild.sh` rebuilds all four production indexes (zoom 8 and 10, CLS
and patch) sequentially with the current preprocessing. The viewer caches fetched tiles under
`outputs/tile_cache/`; set `MURRAY_TILE_CACHE_DIR` to reuse an existing cache.

## Quick start: your own CTX products

For work at native CTX resolution rather than the mosaic:

```bash
# Download -> ISIS3 map-projection -> tile -> embed -> FAISS, one command
pixi run ctx-pipeline --limit 1000

# Estimate disk and time first
pixi run ctx-pipeline --limit 5000 --dry-run

# Serve the local viewer over the result
pixi run ctx-viewer --index-dir outputs/ctx_similarity --source-dir data/raw/ctx

# Legacy Streamlit crop-and-search app
pixi run ctx-similarity-app
```

ISIS3 conflicts with Python 3.12 and lives in its own conda environment. See
[docs/ISIS3_SETUP.md](docs/ISIS3_SETUP.md), [docs/QUICKSTART_ISIS3.md](docs/QUICKSTART_ISIS3.md)
and [docs/CTX_ISIS3_EXAMPLE.md](docs/CTX_ISIS3_EXAMPLE.md).

## Terrain clustering

Groups tile embeddings with HDBSCAN and scores each tile for novelty by kNN distance. Run it
over the embeddings an index build already produced:

```bash
pixi run ctx-terrain --embeddings outputs/ctx_similarity/embeddings.parquet --output outputs/ctx_terrain
```

or as a standalone download-tile-embed-cluster job driven by `configs/ctx_terrain.yaml`:

```bash
pixi run ctx-terrain --images data/raw/ctx
```

Outputs are `tile_clusters.csv` (an integer `cluster_id` per tile, -1 for noise, plus
membership probability and outlier score) and `tile_novelty.csv`. **Clusters are not named.**
Whether cluster 3 is "small fresh craters" is a judgement you make by inspecting it, and the
viewer is the tool for that: pick a tile from the cluster, search for it, and see where it
lands and what its neighbours look like.

## Deploying to Hugging Face Spaces

`Dockerfile` and `app.py` are a Docker-SDK Space entrypoint. The index is pulled from a
companion HF Dataset on cold start rather than baked into the image. See
[docs/HF_SPACES.md](docs/HF_SPACES.md).

## Layout

```
src/ctx_explorer/
├── embeddings/          # DINOv3 extractors (torch.hub + HF), batched embedding pipeline
├── clustering/          # HDBSCAN clusterer, kNN / LOF / centroid novelty detectors
├── terrain/             # Terrain clustering pipeline, tiler, `ctx-terrain-pipeline` CLI
├── downloader.py        # PDS ODE download + ISIS3 processing
├── retrieval.py         # Whole-tile (CLS) FAISS index + chunkwise tiling
├── patch_retrieval.py   # Patch-level index + sliding-window matching
├── atlas.py             # Per-product atlas for the local viewer
├── anomaly.py           # UMAP + HDBSCAN overlay for the local viewer
└── pipeline_runner.py   # Download -> ISIS3 -> tile -> embed -> index driver
scripts/
├── murray_viewer.py, stream_murray_index.py, stream_murray_patch_index.py
├── ctx_viewer_server.py, ctx_similarity_app.py
├── run_ctx_pipeline.py, build_ctx_retrieval_index.py, build_ctx_patch_index.py
├── download_ctx_images.py, setup_isis3.sh, check_tif.py
└── run_aspectpad_rebuild.sh
configs/ctx_terrain.yaml
tests/unit/test_murray_geometry.py
docs/                    # ISIS3 setup, HF Spaces deployment
```

Two tiling implementations still exist: `terrain/tiling.py` (single-scale, quality filtered)
and the multi-scale chunk tiler in `retrieval.py`. Running terrain clustering with
`--embeddings` over an index build sidesteps the older one; folding them together is open.

## Development

```bash
pixi run test
pixi run lint
pixi run format
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
