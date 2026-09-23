# Mars CTX Explorer

Drag a box anywhere on Mars and find the places that look like it.

A similarity-search explorer over Mars Reconnaissance Orbiter CTX imagery and the Murray Lab
global CTX mosaic. Tiles are embedded with DINOv3, indexed with FAISS, and served behind a
Leaflet map. Two viewer backends:

- **Murray viewer** (`scripts/murray_viewer.py`) — fully streamed. The base layer is the Murray
  Lab mosaic via Esri's public ArcGIS tile endpoint. Click or drag-select a region, the backend
  fetches the underlying tiles, embeds them, queries a pre-built global index, and returns
  top-k matches. Supports whole-tile (CLS) and sliding-window patch queries with GPU-vectorised
  template matching.
- **CTX viewer** (`scripts/ctx_viewer_server.py`) — for a locally downloaded set of CTX
  products with tiles on disk. Adds an atlas of unique products and an anomaly overlay.

**Author**: David Ackerman

Split out of the `mars_astrobio` monorepo in September 2026. Siblings: `mars_astrobio`
(CTX terrain clustering) and `backyard_worlds` (brown dwarf detection).

## Quick start: Murray Lab global index

```bash
pixi install
pixi shell

# Build a global CLS index at zoom 8 (streams tiles, no local storage)
pixi run murray-index --zoom 8 --output-dir outputs/murray_z8_cls

# Optional: a patch-level index for sub-tile queries
pixi run murray-patch-index --zoom 8 --output-dir outputs/murray_z8_patch

# Serve the viewer
pixi run murray-viewer --index-dir outputs/murray_z8_cls
```

`scripts/run_aspectpad_rebuild.sh` rebuilds all four production indexes (z8/z10, CLS/patch)
sequentially with the current preprocessing settings.

## Quick start: local CTX products

```bash
# Download → ISIS3 → tile → embed → FAISS in one go
pixi run ctx-pipeline --limit 1000

# Dry run: estimate disk and time first
pixi run ctx-pipeline --limit 5000 --dry-run

# Serve the local viewer
pixi run ctx-viewer --index-dir outputs/ctx_similarity --source-dir data/raw/ctx

# Legacy Streamlit crop-and-search app
pixi run ctx-similarity-app
```

Index artifacts: `embeddings.parquet`, `faiss.index`, `metadata.parquet`, plus a
`*.model.json` sidecar recording the model and preprocessing used to build the index.

ISIS3 conflicts with Python 3.12 and lives in its own conda environment. See
[ISIS3_SETUP.md](ISIS3_SETUP.md) and [QUICKSTART_ISIS3.md](QUICKSTART_ISIS3.md).

## Deploying to Hugging Face Spaces

`Dockerfile` and `app.py` are a Docker-SDK Space entrypoint. The index is pulled from a
companion HF Dataset on cold start rather than baked into the image. See
[docs/HF_SPACES.md](docs/HF_SPACES.md).

## Layout

```
src/ctx_explorer/
├── embeddings/          # DINOv3 extractors (torch.hub + HF), batched embedding pipeline
├── downloader.py        # PDS ODE download + ISIS3 processing
├── retrieval.py         # Whole-image / CLS FAISS index
├── patch_retrieval.py   # Patch-level index + sliding-window matching
├── atlas.py             # Per-product atlas for the local viewer
├── anomaly.py           # UMAP + HDBSCAN anomaly overlay
└── pipeline_runner.py   # Download → ISIS3 → tile → embed → index driver
scripts/
├── murray_viewer.py, stream_murray_index.py, stream_murray_patch_index.py
├── ctx_viewer_server.py, ctx_similarity_app.py
├── run_ctx_pipeline.py, build_ctx_retrieval_index.py, build_ctx_patch_index.py
└── download_ctx_images.py
tests/unit/test_murray_geometry.py
```

## Development

```bash
pixi run test
pixi run lint
pixi run format
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
