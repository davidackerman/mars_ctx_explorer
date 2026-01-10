# ISIS3 Setup Guide for CTX Processing

The CTX downloader now includes full ISIS3 pipeline support for proper geometric correction of Mars Reconnaissance Orbiter CTX images.

## Why ISIS3?

Raw CTX `.IMG` files are unprojected and contain geometric distortions from spacecraft viewing geometry. This causes features like craters to appear **elliptical** instead of circular.

ISIS3 corrects this by:
1. Adding SPICE geometry data (spacecraft position/pointing)
2. Applying radiometric calibration
3. **Map-projecting** to a uniform coordinate system
4. Exporting to standard GeoTIFF format

After processing, craters and other features appear correctly shaped.

## Installation Options

### Option 1: Automated Setup Script (Recommended)

**Important**: ISIS3 requires Python 3.10, but this project uses Python 3.12 in pixi. Therefore, ISIS3 must be installed in a **separate conda environment**.

```bash
# Run the automated setup script
./setup_isis3.sh

# Follow the post-installation steps to activate and configure
```

### Option 2: Manual Conda Installation

```bash
# Create ISIS3 environment with Python 3.10 (separate from pixi)
conda create -y -n isis3 -c conda-forge -c usgs-astrogeology python=3.10 isis

# Activate environment
conda activate isis3

# Download SPICE data for MRO/CTX (required for spiceinit)
# This downloads ~500 MB of spacecraft geometry kernels
downloadIsisData base --mro

# Set ISISROOT (add to ~/.bashrc for persistence)
export ISISROOT=$CONDA_PREFIX
export ISISDATA=$HOME/isis3data  # Or your preferred data location

# Return to base environment (pixi will still find ISIS3 commands in PATH)
conda deactivate
```

**Why a separate environment?** ISIS3 has dependency constraints that conflict with Python 3.12. By keeping it in a separate conda environment, the ISIS3 commands (`mroctx2isis`, `spiceinit`, etc.) remain available in your PATH and the CTX downloader can call them via subprocess.

### Option 3: From Source (Advanced)

See: https://github.com/USGS-Astrogeology/ISIS3

### Option 4: Use GDAL Only (No Geometric Correction)

If you don't need geometric correction, disable ISIS3 in the config:

```yaml
# configs/pipelines/ctx_terrain.yaml
download:
  isis3:
    enabled: false  # Use GDAL for simple format conversion
```

**Warning**: Images will remain unprojected and features may look distorted.

## Testing ISIS3 Installation

```bash
# Check ISIS3 commands are available
which mroctx2isis
which spiceinit
which ctxcal
which cam2map
which isis2std

# Verify ISISROOT is set
echo $ISISROOT

# Run test download
pixi run python test_ctx_download.py
```

## Usage

Once ISIS3 is installed, the CTX downloader will automatically:

```python
from scientific_pipelines.planetary.mars.ctx.downloader import CTXDownloader

# Create downloader (ISIS3 enabled by default)
downloader = CTXDownloader(output_dir="data/raw/ctx")

# Download and process images
image_list = downloader.search_images(limit=10)
tif_paths = downloader.download_images(image_list)
# -> Returns map-projected GeoTIFFs with corrected geometry
```

## Configuration Options

```yaml
# configs/pipelines/ctx_terrain.yaml
download:
  isis3:
    enabled: true                    # Use ISIS3 pipeline
    apply_calibration: true          # Apply ctxcal radiometric calibration
    map_projection: "sinusoidal"     # Map projection (sinusoidal, equirectangular, polarstereographic)
    map_resolution: null             # Output resolution in m/pixel (null = auto ~6 m/pixel)
```

## Pipeline Steps

When ISIS3 is enabled, each image goes through:

1. **mroctx2isis** - Import CTX .IMG to ISIS cube format
2. **spiceinit** - Add SPICE geometry (requires ISISDATA)
3. **ctxcal** - Radiometric calibration (optional)
4. **cam2map** - Map-project to uniform grid ✨ **This fixes elliptical craters!**
5. **isis2std** - Export to GeoTIFF

Intermediate `.cub` files are automatically cleaned up to save disk space.

## Troubleshooting

### "ISIS3 command not found"

- Ensure ISIS3 is installed and in your PATH
- Set `ISISROOT` environment variable
- Activate the isis3 conda environment

### "SPICE data not found"

- Run `downloadIsisData base --mro` to get MRO SPICE kernels
- Verify `ISISDATA` points to the correct location

### "cam2map failed"

- Check the map projection name is valid (sinusoidal, equirectangular, etc.)
- Ensure SPICE kernels cover the image acquisition date

## References

- ISIS3 Documentation: https://isis.astrogeology.usgs.gov/
- CTX Instrument: https://mars.nasa.gov/mro/mission/instruments/ctx/
- SPICE System: https://naif.jpl.nasa.gov/naif/
