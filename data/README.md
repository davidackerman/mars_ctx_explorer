# Data Directory

This directory contains Mars rover data downloaded from NASA's Planetary Data System (PDS).

## Structure

- `raw/` - Raw downloaded data from PDS (gitignored)
  - `watson/` - WATSON micro-imager images organized by sol
  - `sherloc/` - SHERLOC Raman spectroscopy data (future)
  - `pixl/` - PIXL XRF elemental maps (future)

- `processed/` - Preprocessed datasets ready for training (gitignored)
  - Train/val/test splits
  - Organized by texture class

- `annotations/` - Manual annotations and labels (gitignored except schema)
  - Known biosignature sites (Cheyava Falls, Wildcat Ridge)
  - Annotation schema definitions

- `cache/` - HTTP cache for API responses (gitignored)

## Data Organization

### WATSON Images

```
raw/watson/
├── sol_0001/
│   ├── NLF_0001_0000000001_001.IMG   # Raw image file
│   ├── NLF_0001_0000000001_001.xml   # PDS4 label (metadata)
│   └── metadata.json                  # Parsed metadata
├── sol_0002/
└── manifest.json                      # Download tracking
```

## Getting Data

To download WATSON images:

```bash
# Download first 100 sols
pixi run download-watson

# Download specific sol range
python scripts/download_data.py --instrument watson --sols 500-600

# Download with NASA API key
export NASA_API_KEY="your_key_here"
python scripts/download_data.py --instrument watson --sols 0-100
```

## Data Sources

- **PDS Geosciences Node**: https://pds-geosciences.wustl.edu/missions/mars2020/
- **NASA Mars Photos API**: https://api.nasa.gov/mars-photos/api/v1
- **API Key**: Get your free key at https://api.nasa.gov/

## Notes

- Raw data files are large (several GB) and are gitignored
- Keep only the manifest and README files in version control
- Use the cache directory to avoid re-downloading data
