# Quick Start: ISIS3 for CTX Processing

## TL;DR

```bash
# 1. Run setup script
./setup_isis3.sh

# 2. Activate ISIS3 environment and download SPICE data
conda activate isis3
downloadIsisData base --mro
export ISISROOT=$CONDA_PREFIX
export ISISDATA=$HOME/isis3data

# 3. Return to base environment
conda deactivate

# 4. Test CTX download with ISIS3 processing
pixi run python test_ctx_download.py
```

## What This Does

The setup creates a **separate conda environment** for ISIS3 because:
- **This project**: Uses Python 3.12 (modern features, better performance)
- **ISIS3**: Requires Python 3.10 (dependency constraints from legacy libs)

Your CTX downloader runs in the pixi environment (Python 3.12) and calls ISIS3 commands (from Python 3.10 environment) via subprocess. This is **the standard workflow** for ISIS3 integration.

## Verifying Installation

```bash
# Check ISIS3 is in PATH
which mroctx2isis
# Should output: /path/to/conda/envs/isis3/bin/mroctx2isis

# Check environment variables
echo $ISISROOT   # Should point to isis3 conda env
echo $ISISDATA   # Should point to SPICE data directory

# Test with example image
rm -rf data/test/ctx_single/*
pixi run python test_ctx_download.py
```

## Expected Output

**Without ISIS3** (before setup):
```
[1/5] mroctx2isis: image.IMG
ERROR: ISIS3 command not found
WARNING: Falling back to GDAL conversion
INFO: Converted to GeoTIFF (GDAL - no geometry correction)
```

**With ISIS3** (after setup):
```
[1/5] mroctx2isis: image.IMG
[2/5] spiceinit: image.cub
[3/5] ctxcal: image.cub
[4/5] cam2map: image.cal.cub
[5/5] isis2std: image.map.cub -> image.tif
INFO: Successfully processed through ISIS3 pipeline
```

## Troubleshooting

### "downloadIsisData: command not found"
Activate the isis3 environment first:
```bash
conda activate isis3
downloadIsisData base --mro
```

### "SPICE data not found" during spiceinit
Set ISISDATA environment variable:
```bash
export ISISDATA=$HOME/isis3data
# Add to ~/.bashrc for persistence
```

### ISIS3 commands not found when running pipeline
Make sure the isis3 environment is added to your PATH. You can do this by:
```bash
# Add to ~/.bashrc
conda activate isis3
conda deactivate
# This initializes conda and adds all envs to PATH
```

Or use absolute paths in the downloader (not recommended).

## File Sizes

- **ISIS3 conda package**: ~2 GB
- **MRO SPICE data**: ~500 MB
- **Example CTX image**:
  - Raw .IMG: 252 MB
  - GDAL GeoTIFF: 79 MB (no geometric correction)
  - ISIS3 GeoTIFF: ~80 MB (with geometric correction)

## Next Steps

See [ISIS3_SETUP.md](ISIS3_SETUP.md) for detailed configuration options and [CTX_ISIS3_EXAMPLE.md](CTX_ISIS3_EXAMPLE.md) for test results.
