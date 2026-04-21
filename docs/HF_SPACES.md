# Deploying the CTX similarity viewer to Hugging Face Spaces

The `Dockerfile` + `app.py` in the repo root are a HF-Spaces-compatible
entrypoint. The index itself is **not** baked into the image — it's pulled
from a companion HF Dataset on cold start. Keep the Dockerfile layers stable
and push updated indexes by re-uploading to the dataset (cheap).

## One-time setup

1. **Create two HF repos**
   - A **Space** (Docker SDK) for the viewer — e.g. `yourname/ctx-similarity`.
   - A **Dataset** for the index artefacts — e.g. `yourname/ctx-similarity-index`.

2. **Push the viewer Space**
   ```bash
   git clone https://huggingface.co/spaces/yourname/ctx-similarity
   cd ctx-similarity
   # Copy repo contents into the Space's git working tree
   cp -r ../mars_astrobio/Dockerfile ../mars_astrobio/app.py \
         ../mars_astrobio/src ../mars_astrobio/scripts \
         ../mars_astrobio/pyproject.toml .
   git add . && git commit -m "Initial CTX viewer" && git push
   ```

3. **Upload the index to the Dataset**
   Use `huggingface-cli upload` or the Python `HfApi.upload_folder`:
   ```bash
   huggingface-cli login   # once, with a write token
   huggingface-cli upload \
       yourname/ctx-similarity-index \
       outputs/ctx_similarity/ \
       . \
       --repo-type=dataset \
       --include="faiss.index" "metadata.parquet" \
                 "faiss.model.json" "anomaly_scores.parquet" \
                 "atlas.parquet" "embeddings.parquet"
   ```
   *Don't* upload the `tiles/` directory or the full-resolution source `.tifs`
   — those are huge and the viewer doesn't need them at query time (see the
   SOURCE_DIR story below).

4. **Configure the Space**
   In the Space settings, set:
   - `INDEX_REPO_ID=yourname/ctx-similarity-index`
   - Enable the **ZeroGPU** hardware pool for on-demand GPU inference. The
     Space sleeps when idle (no cost) and a GPU is attached briefly on query.

## Source-tile fetching

The viewer's `/api/source` endpoint reads the full-resolution `.tif` for the
selected product. On HF Spaces we don't keep those — they're 50 MB each. Two
options:

- **`SOURCE_DIR` on a persistent volume** (paid tier) — `/data/source`.
- **Rewrite `/api/source` to lazily pull from the ODE public mirror** and
  cache in `/tmp`. For the demo, this is the honest path — source tiles are
  public. Implementation: change `api_source` to fall back to fetching
  `https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/.../{product_id}.IMG`
  then passing it through the same ISIS3 pipeline. ISIS3 in a Docker image
  is non-trivial — an alternative is to precompute downsampled JPEG previews
  per product and upload them to the same HF Dataset (say `/previews/{product_id}.jpg`).

For a first demo, option 2b (pre-rendered preview JPEGs in the dataset) is the
simplest — our existing `api_source` already produces downsampled JPEG
previews, so we can just save those once before the HF upload.

## What the HF-hosted viewer exposes

- `/` — Leaflet map + discovery sidebar.
- `/api/images` — product list with approximate lat/lon.
- `/api/tile`, `/api/source` — streams image assets (see caveats above).
- `/api/query` — click-to-similar.
- `/api/anomalies` — "surprise me" button.
- `/api/few_shot` — positive-tray search.
- `/api/atlas` — UMAP terrain atlas scatter.
- `/api/build_patch_index` — lazy per-product patch index (heavy; only
  interesting when a user wants region-level queries on a specific product).

## Cost

- **ZeroGPU**: free on the community tier, rate-limited. DINO ViT-L forward
  takes ~300 ms on an A10G; query end-to-end ~1–2 s.
- **CPU-only** tier: free but slower (~2 s DINO forward on CPU, ~5 s
  end-to-end).
- **Persistent storage**: the index + previews will take 1–5 GB depending
  on corpus size. HF Datasets are free up to ~50 GB.
