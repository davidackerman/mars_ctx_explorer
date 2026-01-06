# Mars Rover Biosignature Detection

AI/ML-based detection of life-like signatures from Mars rover data using deep learning on WATSON micro-imager textures.

## Overview

This repository implements a PyTorch-based pipeline for detecting biosignature-proxy anomalies in Mars Perseverance rover imagery. Starting with WATSON micro-imager texture analysis, the system can identify patterns like:

- **Fine laminations** and micro-porosity indicating sedimentary processes
- **Dissolution cavities** suggesting water-rock interactions
- **Iron oxide halos** and bleached spots (Cheyava Falls-like features)
- **Chemical alteration textures** potentially linked to biological activity

The approach uses deep learning to scan thousands of Mars rover images and identify regions with biosignature-like characteristics, helping prioritize targets for detailed scientific analysis.

## Features

- **Automated PDS Pipeline**: Download and parse WATSON images from NASA's Planetary Data System
- **Deep Learning Models**: CNN and Vision Transformer architectures for texture classification
- **Biosignature Focus**: Fine-tuned on known sites (Cheyava Falls, Wildcat Ridge)
- **Extensible Architecture**: Ready for SHERLOC/PIXL multi-modal integration
- **Production Ready**: Pixi-managed dependencies, comprehensive testing, reproducible workflows

## Quick Start

### Prerequisites

- [Pixi](https://pixi.sh) package manager
- CUDA-capable GPU (recommended for training)
- NASA API key (get one at https://api.nasa.gov/)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mars_astrobio.git
cd mars_astrobio

# Install dependencies with Pixi
pixi install

# Activate environment
pixi shell

# Set up NASA API key in .env file
# Open .env and add: NASA_API_KEY=your_actual_key_here
# See SETUP_API_KEY.md for detailed instructions
```

### Usage

```bash
# Download WATSON data for first 100 sols
pixi run download-watson

# Prepare dataset (split into train/val/test)
pixi run prepare-data

# Train baseline model
pixi run train

# Run inference on entire mission archive
python scripts/predict.py --checkpoint models/production/watson_texture_v1.pt

# Explore results in Jupyter notebooks
pixi run notebook
```

## Project Structure

```
mars_astrobio/
├── src/mars_biosig/        # Main package
│   ├── data/               # Data pipeline (PDS client, downloaders, datasets)
│   ├── models/             # ML architectures (CNN, ViT)
│   ├── training/           # Training infrastructure
│   ├── inference/          # Prediction and visualization
│   └── multimodal/         # Future: multi-sensor fusion
├── configs/                # YAML configurations
├── scripts/                # CLI tools (download, train, predict)
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit and integration tests
├── data/                   # Downloaded and processed data (gitignored)
└── models/                 # Trained models (gitignored)
```

See [the implementation plan](/.claude/plans/elegant-questing-salamander.md) for detailed architecture documentation.

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
- [ ] Phase 2: Data Pipeline - PDS client and WATSON downloader
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
  author = {Ackerman, Daniel},
  year = {2025},
  url = {https://github.com/yourusername/mars_astrobio}
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
