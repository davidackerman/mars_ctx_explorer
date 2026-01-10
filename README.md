# Scientific Pipelines: Multi-Domain Unsupervised ML

Unsupervised machine learning pipelines for planetary and astronomical data analysis using self-supervised vision transformers and density-based clustering.

**Author**: David Ackerman

**Current Status**: Multi-domain architecture with two operational pipelines:
1. **Mars CTX Terrain Classification** - Unsupervised clustering of Mars orbital imagery
2. **Backyard Worlds Brown Dwarf Detection** - Automated candidate ranking from Zooniverse flipbooks

## Overview

This repository implements end-to-end unsupervised ML pipelines for scientific discovery across planetary science and astronomy. Using state-of-the-art self-supervised embeddings (DINOv3) and density-based clustering (HDBSCAN), the pipelines automatically categorize data and identify novel/anomalous regions without human labels.

### Mars CTX Terrain Classification

Automatically cluster Mars Reconnaissance Orbiter Context Camera (CTX) images into terrain categories and identify geologically unique regions:

- **Unsupervised terrain categories** (craters, channels, dunes, plains, etc.)
- **Novelty detection** for unusual/interesting geological features
- **No human labels required** - fully automated discovery
- Processes orbital imagery at ~6 m/pixel resolution

### Backyard Worlds Brown Dwarf Detection

Fully automated AI for ranking brown dwarf candidates in WISE infrared flipbook sequences:

- **Motion analysis** - slow proper motion detection
- **Multi-factor scoring** - morphology, color, novelty, artifacts
- **Behavior clustering** - stationary stars, fast movers, slow movers
- **No citizen labels needed** - replaces manual Zooniverse classification

## Features

### Core Infrastructure
- **DINOv3 Vision Transformer**: State-of-the-art self-supervised embeddings (768-dim)
- **HDBSCAN Clustering**: Automatic cluster detection with outlier handling
- **Novelty Detection**: FAISS-accelerated kNN for fast anomaly scoring
- **Batch Processing**: Resume-capable pipelines with progress tracking
- **Multi-Domain**: Unified architecture for planetary and astronomy domains

### Mars CTX Pipeline
- **Image Tiling**: 256×256 patches with quality filtering (contrast, data fraction)
- **Unsupervised Clustering**: Automatic terrain categorization
- **Novelty Scoring**: Identify unique/interesting geological regions
- **Outputs**: CSV files + Parquet embeddings for downstream analysis

### Backyard Worlds Pipeline
- **Zooniverse Integration**: Panoptes API for flipbook download
- **Temporal Encoding**: Motion-aware sequence embeddings
- **Brown Dwarf Scoring**: Multi-factor heuristics (motion, morphology, color, novelty)
- **Behavior Clustering**: Automatic classification of object types

### Production Ready
- **Pixi-managed dependencies**: Reproducible environments
- **Comprehensive logging**: Track pipeline progress
- **Resume capability**: Continue interrupted runs
- **Modular design**: Easy to extend with new pipelines

## Quick Start

### Prerequisites

- [Pixi](https://pixi.sh) package manager
- CUDA-capable GPU (recommended for training)
- NASA API key (get one at https://api.nasa.gov/)

### Installation

```bash
# Clone repository
git clone https://github.com/davidackerman/mars_astrobio.git
cd mars_astrobio

# Install dependencies with Pixi
pixi install

# Activate environment
pixi shell

# Set up NASA API key in .env file (optional, for legacy WATSON pipeline)
# Open .env and add: NASA_API_KEY=your_actual_key_here
```

### Usage

#### Mars CTX Terrain Classification

```bash
# Run full pipeline (assumes CTX images in data/raw/ctx/)
pixi run ctx-pipeline

# Or use the CLI directly
ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --images data/raw/ctx/

# Resume from previous run
ctx-terrain-pipeline --config configs/pipelines/ctx_terrain.yaml --skip-tiling --skip-embedding
```

**Outputs** (in `outputs/ctx_terrain/`):
- `tiles.csv` - Tile metadata with quality metrics
- `embeddings.parquet` - 768-dim DINOv3 embeddings
- `tile_clusters.csv` - Cluster assignments and probabilities
- `tile_novelty.csv` - Novelty scores for unique regions

#### Backyard Worlds Brown Dwarf Detection

```bash
# Run full pipeline (downloads from Zooniverse)
pixi run backyard-worlds

# Or use the CLI directly
backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml

# With authentication for higher rate limits
backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml \
  --username myuser --password mypass

# Resume from previous run
backyard-worlds-pipeline --config configs/pipelines/backyard_worlds.yaml --skip-download --skip-encoding
```

**Outputs** (in `outputs/backyard_worlds/`):
- `subjects.csv` - Subject metadata
- `embeddings.parquet` - 2304-dim sequence embeddings
- `subject_clusters.csv` - Behavior cluster assignments
- `brown_dwarf_ranking.csv` - Ranked candidates by score

#### Legacy WATSON Pipeline

```bash
# Original WATSON texture classification (still supported)
pixi run download-watson
pixi run train
```

## Project Structure

```
scientific_pipelines/
├── src/scientific_pipelines/  # Main package
│   ├── core/                  # Shared infrastructure
│   │   ├── embeddings/        # DINOv3, SimCLR extractors
│   │   ├── clustering/        # HDBSCAN, novelty detection
│   │   ├── data/              # Base dataset classes
│   │   ├── training/          # Training infrastructure
│   │   └── utils/             # Config, logging
│   ├── planetary/mars/        # Planetary science domain
│   │   ├── watson/            # WATSON texture classification (legacy)
│   │   └── ctx/               # CTX terrain classification
│   │       ├── tiling.py      # Image tiling with quality filters
│   │       ├── pipeline.py    # End-to-end orchestration
│   │       └── scripts.py     # CLI entry point
│   └── astronomy/             # Astronomy domain
│       └── backyard_worlds/   # Brown dwarf detection
│           ├── downloader.py  # Panoptes API integration
│           ├── sequence_encoder.py  # Flipbook encoding
│           ├── brown_dwarf_scorer.py  # Multi-factor scoring
│           ├── pipeline.py    # End-to-end orchestration
│           └── scripts.py     # CLI entry point
├── configs/pipelines/         # Pipeline configurations
│   ├── ctx_terrain.yaml       # CTX configuration
│   └── backyard_worlds.yaml   # Backyard Worlds configuration
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit and integration tests
├── data/                      # Downloaded and processed data (gitignored)
├── outputs/                   # Pipeline outputs (gitignored)
└── models/                    # Trained models (gitignored)
```


## Scientific Background

### Target Features

The system focuses on detecting micro-textures associated with potential biosignatures:

1. **Laminations**: Fine-scale layering in sedimentary rocks, potentially indicating cyclical depositional processes
2. **Micro-porosity**: Small voids and cavities that could have hosted microbial communities
3. **Redox Features**: Iron oxide halos and color variations suggesting redox boundaries
4. **Alteration Zones**: Chemically altered regions like the "leopard spots" in Cheyava Falls

### Known Biosignature Sites

Training data includes labeled examples from:
- **Cheyava Falls** (Sol 1174): Leopard-spot patterns with organic signatures and redox minerals
- **Wildcat Ridge** (Sol 528): Strongest organic detection in Jezero Crater
- **Delta deposits**: Sedimentary structures in the ancient river delta

## Development

### Setting Up Development Environment

```bash
# Install with dev dependencies
pixi install

# Install pre-commit hooks
pixi run pre-commit install

# Run tests
pixi run test

# Format code
pixi run format

# Lint code
pixi run lint

# Type check
pixi run typecheck
```

### Running Tests

```bash
# All tests
pixi run test

# Specific test file
pytest tests/unit/test_pds_client.py -v

# With coverage report
pytest --cov=mars_biosig --cov-report=html
```

## Data Access

### NASA PDS Data Sources

- **PDS Geosciences Node**: https://pds-geosciences.wustl.edu/missions/mars2020/
- **Mars Photos API**: https://api.nasa.gov/mars-photos/api/v1
- **WATSON Camera**: Perseverance rover's micro-imager instrument

### Instruments

- **WATSON** (Wide Angle Topographic Sensor for Operations and eNgineering): High-resolution micro-imaging camera
- **SHERLOC** (Future): Raman spectroscopy for molecular composition
- **PIXL** (Future): X-ray fluorescence for elemental mapping

## Roadmap

- [x] Phase 1: Foundation - Repository structure and configuration
- [x] Phase 2: Data Pipeline - PDS client and WATSON downloader (**Currently Complete**)
- [ ] Phase 3: Model Architecture - Texture CNN baseline
- [ ] Phase 4: Training Pipeline - Training loop and evaluation
- [ ] Phase 5: Inference & Analysis - Mission-wide scanning
- [ ] Phase 6: Multi-Modal Integration - SHERLOC + PIXL fusion

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mars_biosig2025,
  title = {Mars Rover Biosignature Detection},
  author = {Ackerman, David},
  year = {2025},
  url = {https://github.com/davidackerman/mars_astrobio}
}
```

## Related Work

- Cheyava Falls biosignature discovery: [Nature 2025](https://www.nature.com/articles/s41586-025-09413-0)
- Mars 2020 Mission: [NASA JPL](https://mars.nasa.gov/mars2020/)
- WATSON Camera: [Malin Space Science Systems](https://www.msss.com/)

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA JPL Mars 2020 Mission team
- PDS Geosciences Node for data access
- ChatGPT for initial project planning and scientific context
