# Dockerfile for deploying the CTX similarity viewer to Hugging Face Spaces
# (Docker SDK) or any container host.
#
# The index + metadata aren't baked into the image — they're pulled from a
# companion HF Dataset at runtime via HF_HUB_REPO_ID. The DINO weights also
# stream from HF Hub on cold start.

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps: GDAL for reading the 16-bit CTX TIFFs; rclone for fetching
# index artefacts; git for huggingface-hub auto-install niceties.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev python3-gdal \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expects port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf_cache
ENV HF_HUB_CACHE=/data/hf_cache/hub
ENV TRANSFORMERS_CACHE=/data/hf_cache/transformers

# Deps. We keep this lean (no pixi) — HF Spaces cold-start is bounded by pip.
RUN pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" \
        "transformers>=4.56.0,<5.0" "huggingface_hub>=0.34.0,<1.0" \
        "torchvision>=0.15.0" \
        "pandas>=2.2.0" "numpy>=1.26.0,<2.4" "pyarrow>=14.0.0" \
        "pillow>=10.0.0" "scipy>=1.12.0" "scikit-learn>=1.4.0" \
        "umap-learn>=0.5.5" "hdbscan>=0.8.33" \
        "faiss-cpu>=1.7.4" "requests>=2.31.0" "tqdm>=4.66.0" \
        "streamlit-cropper>=0.3.1"

WORKDIR /app
COPY src /app/src
COPY scripts /app/scripts
COPY app.py /app/app.py
COPY pyproject.toml /app/pyproject.toml

ENV PYTHONPATH=/app/src:${PYTHONPATH}

# Default: pull index from the HF dataset repo, then start the viewer.
# Override INDEX_REPO_ID via Space environment to point at a different
# dataset (e.g. your own fork).
ENV INDEX_REPO_ID="davidackerman/ctx-similarity-index"
ENV INDEX_LOCAL_DIR=/data/index
ENV SOURCE_DIR=/data/source

CMD ["python", "app.py"]
