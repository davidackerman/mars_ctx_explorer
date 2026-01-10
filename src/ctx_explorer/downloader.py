"""Download CTX images from NASA PDS Orbital Data Explorer."""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CTXDownloader:
    """
    Download CTX images from NASA PDS Orbital Data Explorer.

    CTX (Context Camera) captures wide-area imaging of Mars surface
    from the Mars Reconnaissance Orbiter at ~6 m/pixel resolution.

    This downloader uses the PDS Geosciences Node ODE REST API to search
    for and download CTX EDR (Experiment Data Records) images.

    Downloaded images are automatically processed through the ISIS3 pipeline
    to correct spacecraft geometry and map-project to a uniform ground grid.
    This ensures features like craters appear circular rather than elliptical.

    ISIS3 Processing Pipeline:
        1. mroctx2isis - Import CTX .IMG to ISIS cube format
        2. spiceinit - Add SPICE geometry (spacecraft position/pointing)
        3. ctxcal - Radiometric calibration (optional)
        4. cam2map - Map-project to uniform coordinate system
        5. isis2std - Export to GeoTIFF

    The original IMG files are removed after conversion to save disk space.

    Args:
        output_dir: Directory to save downloaded images
        api_base_url: ODE REST API endpoint (default: oderest.rsl.wustl.edu)
        max_retries: Maximum number of download retries (default: 3)
        retry_delay: Delay between retries in seconds (default: 5.0)
        use_isis3: Use ISIS3 pipeline for geometric correction (default: True)
        apply_calibration: Apply radiometric calibration with ctxcal (default: True)
        map_projection: Map projection for cam2map (default: "sinusoidal")
            Options: sinusoidal, equirectangular, polarstereographic
        map_resolution: Output resolution in m/pixel (default: None = auto ~6m/px)

    Example:
        >>> # Download with ISIS3 processing (recommended)
        >>> downloader = CTXDownloader(output_dir="data/raw/ctx")
        >>> # Search for 100 images
        >>> image_list = downloader.search_images(limit=100)
        >>> # Download, process through ISIS3, and export to GeoTIFF
        >>> paths = downloader.download_images(image_list)
        >>>
        >>> # Or download specific region (Jezero crater)
        >>> paths = downloader.download_region(
        ...     min_lon=-77.5, max_lon=-77.0,
        ...     min_lat=18.0, max_lat=18.5,
        ...     limit=50
        ... )
        >>>
        >>> # For quick testing without ISIS3 (no geometric correction)
        >>> downloader_fast = CTXDownloader(
        ...     output_dir="data/raw/ctx",
        ...     use_isis3=False
        ... )
    """

    # ODE REST API constants
    DEFAULT_API_BASE = "https://oderest.rsl.wustl.edu/live2/"  # Note: trailing slash required
    INSTRUMENT_HOST_ID = "MRO"  # Mars Reconnaissance Orbiter
    INSTRUMENT_ID = "CTX"  # Context Camera
    PRODUCT_TYPE = "EDR"  # Experiment Data Record

    def __init__(
        self,
        output_dir: Path,
        api_base_url: str = DEFAULT_API_BASE,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        use_isis3: bool = True,
        apply_calibration: bool = True,
        map_projection: str = "sinusoidal",
        map_resolution: Optional[float] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_base_url = api_base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ISIS3 processing options
        self.use_isis3 = use_isis3
        self.apply_calibration = apply_calibration
        self.map_projection = map_projection
        self.map_resolution = map_resolution  # m/pixel, None = auto (native ~6m/pixel)

        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

        logger.info(f"CTXDownloader initialized with output_dir={self.output_dir}")
        logger.info(f"  ISIS3 processing: {'enabled' if use_isis3 else 'disabled'}")
        if use_isis3:
            logger.info(f"  Calibration: {'enabled' if apply_calibration else 'disabled'}")
            logger.info(f"  Map projection: {map_projection}")
            logger.info(f"  Resolution: {map_resolution or 'auto'} m/pixel")

    def search_images(
        self,
        limit: int = 100,
        min_lon: Optional[float] = None,
        max_lon: Optional[float] = None,
        min_lat: Optional[float] = None,
        max_lat: Optional[float] = None,
    ) -> List[Dict]:
        """
        Search for CTX images using the ODE REST API.

        Args:
            limit: Maximum number of images to return
            min_lon: Western longitude bound (degrees, -180 to 180)
            max_lon: Eastern longitude bound (degrees, -180 to 180)
            min_lat: Southern latitude bound (degrees, -90 to 90)
            max_lat: Northern latitude bound (degrees, -90 to 90)

        Returns:
            List of image metadata dictionaries with ProductId, URLs, coordinates

        Example:
            >>> # Search for 50 images anywhere on Mars
            >>> images = downloader.search_images(limit=50)
            >>> # Search in Jezero crater region
            >>> images = downloader.search_images(
            ...     limit=100,
            ...     min_lon=-77.5, max_lon=-77.0,
            ...     min_lat=18.0, max_lat=18.5
            ... )
        """
        logger.info(f"Searching for up to {limit} CTX images via ODE API")

        params = {
            "query": "product",
            "results": "fmpc",  # Return files, metadata, products, count
            "output": "JSON",
            "target": "Mars",
            "ihid": self.INSTRUMENT_HOST_ID,
            "iid": self.INSTRUMENT_ID,
            "pt": self.PRODUCT_TYPE,
            "limit": limit,
            "offset": 0,
        }

        # Add geographic bounds if specified
        if min_lon is not None:
            params["westernlon"] = min_lon
            logger.info(f"  Filtering: western longitude >= {min_lon}")
        if max_lon is not None:
            params["easternlon"] = max_lon
            logger.info(f"  Filtering: eastern longitude <= {max_lon}")
        if min_lat is not None:
            params["minlat"] = min_lat
            logger.info(f"  Filtering: latitude >= {min_lat}")
        if max_lat is not None:
            params["maxlat"] = max_lat
            logger.info(f"  Filtering: latitude <= {max_lat}")

        try:
            response = requests.get(self.api_base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to query ODE API: {e}")
            raise

        data = response.json()
        products = self._parse_ode_response(data)

        logger.info(f"Found {len(products)} CTX images matching search criteria")
        return products

    def download_images(
        self,
        image_list: List[Dict],
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Download CTX image files from ODE search results and process them.

        Downloads IMG (image data) and XML (PDS label) files for each product,
        then processes through ISIS3 pipeline (if enabled) or GDAL conversion.

        Args:
            image_list: List of image metadata dicts from search_images()
            overwrite: If True, re-download and reprocess existing files

        Returns:
            List of paths to processed GeoTIFF files

        Example:
            >>> images = downloader.search_images(limit=10)
            >>> tif_paths = downloader.download_images(images)
        """
        logger.info(f"Downloading {len(image_list)} CTX images")

        downloaded_paths = []

        for image_meta in tqdm(image_list, desc="Downloading CTX images"):
            product_id = image_meta["product_id"]

            # Check if already downloaded (resume capability)
            if not overwrite and product_id in self.manifest["downloaded_images"]:
                existing = self.manifest["downloaded_images"][product_id]
                img_path = Path(existing["img_path"])
                if img_path.exists():
                    logger.debug(f"Skipping {product_id} (already downloaded)")
                    downloaded_paths.append(img_path)
                    continue

            # Download IMG and XML files
            try:
                img_path = self._download_product_files(image_meta)
                downloaded_paths.append(img_path)

                # Update manifest
                self._add_to_manifest(image_meta, img_path)

            except Exception as e:
                logger.error(f"Failed to download {product_id}: {e}")
                continue

        self._save_manifest()
        logger.info(f"Successfully downloaded {len(downloaded_paths)} images")

        return downloaded_paths

    def download_region(
        self,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        limit: int = 100,
    ) -> List[Path]:
        """
        Download and process all CTX images covering a geographic region.

        Convenience method that combines search_images() and download_images().
        Images are processed through ISIS3 pipeline if enabled.

        Args:
            min_lon: Western longitude bound (degrees, -180 to 180)
            max_lon: Eastern longitude bound (degrees, -180 to 180)
            min_lat: Southern latitude bound (degrees, -90 to 90)
            max_lat: Northern latitude bound (degrees, -90 to 90)
            limit: Maximum number of images to download

        Returns:
            List of paths to processed GeoTIFF files

        Example:
            >>> # Download and map-project images from Jezero crater
            >>> paths = downloader.download_region(
            ...     min_lon=-77.5, max_lon=-77.0,
            ...     min_lat=18.0, max_lat=18.5,
            ...     limit=50
            ... )
        """
        logger.info(
            f"Downloading CTX images for region: "
            f"lon=[{min_lon}, {max_lon}], lat=[{min_lat}, {max_lat}]"
        )

        image_list = self.search_images(
            limit=limit,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat,
        )

        if not image_list:
            logger.warning("No images found in specified region")
            return []

        return self.download_images(image_list)

    def _parse_ode_response(self, data: Dict) -> List[Dict]:
        """
        Parse ODE REST API JSON response to extract product metadata.

        Args:
            data: JSON response from ODE API

        Returns:
            List of product metadata dictionaries
        """
        products = []

        # ODE API returns products in 'ODEResults' -> 'Products' -> 'Product'
        try:
            ode_results = data.get("ODEResults", {})
            products_data = ode_results.get("Products", {})
            product_list = products_data.get("Product", [])

            # Handle single product (not in list)
            if isinstance(product_list, dict):
                product_list = [product_list]

            for product in product_list:
                product_id = product.get("pdsid") or product.get("ProductId")
                if not product_id:
                    logger.warning("Product missing ID, skipping")
                    continue

                # Extract product file URLs
                product_files = product.get("Product_files", {}).get("Product_file", [])
                if isinstance(product_files, dict):
                    product_files = [product_files]

                # Debug: log if no product files found
                if not product_files:
                    logger.debug(f"Product {product_id} has no Product_files in response")
                    # Try to construct URL from LabelURL if available
                    label_url = product.get("LabelURL")
                    if label_url:
                        img_url = label_url
                        xml_url = None
                    else:
                        logger.warning(f"Product {product_id} missing file URLs, skipping")
                        continue
                else:
                    # Find IMG and XML URLs
                    img_url = None
                    xml_url = None
                    for pfile in product_files:
                        url = pfile.get("URL", "")
                        ptype = pfile.get("Type", "").upper()
                        filename = pfile.get("FileName", "").upper()

                        # Look for .IMG files (Type is "Product" for IMG files in CTX EDR)
                        if filename.endswith(".IMG") and not filename.endswith(".JPEG"):
                            img_url = url
                        # XML labels have Type="Label" or end with .XML
                        elif ptype == "LABEL" or url.upper().endswith(".XML"):
                            xml_url = url

                    if not img_url:
                        logger.warning(f"Product {product_id} missing IMG URL, skipping")
                        continue

                # Extract metadata
                products.append({
                    "product_id": product_id,
                    "img_url": img_url,
                    "xml_url": xml_url,
                    "center_lon": product.get("CenterLongitude"),
                    "center_lat": product.get("CenterLatitude"),
                    "emission_angle": product.get("EmissionAngle"),
                    "incidence_angle": product.get("IncidenceAngle"),
                    "solar_longitude": product.get("SolarLongitude"),
                })

        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse ODE response: {e}")
            logger.debug(f"Response data: {data}")

        return products

    def _download_product_files(self, image_meta: Dict) -> Path:
        """
        Download IMG and XML files for a CTX product, then convert to GeoTIFF.

        Args:
            image_meta: Product metadata dict with URLs

        Returns:
            Path to converted GeoTIFF file (or IMG file if conversion fails)
        """
        product_id = image_meta["product_id"]
        img_url = image_meta["img_url"]
        xml_url = image_meta.get("xml_url")

        # Download IMG file
        img_filename = f"{product_id}.IMG"
        img_path = self.output_dir / img_filename
        self._download_file(img_url, img_path)

        # Download XML label if available
        if xml_url:
            xml_filename = f"{product_id}.xml"
            xml_path = self.output_dir / xml_filename
            try:
                self._download_file(xml_url, xml_path)
            except Exception as e:
                logger.warning(f"Failed to download XML label for {product_id}: {e}")

        # Process IMG through ISIS3 pipeline or convert with GDAL
        tif_path = None
        if self.use_isis3:
            tif_path = self._process_ctx_isis3(img_path)
            # If ISIS3 fails, try GDAL fallback
            if tif_path is None and img_path.exists():
                logger.warning(
                    f"ISIS3 processing failed for {product_id}, falling back to GDAL conversion"
                )
                tif_path = self._convert_img_to_tif_gdal(img_path)
        else:
            tif_path = self._convert_img_to_tif_gdal(img_path)

        return tif_path if tif_path else img_path

    def _download_file(self, url: str, output_path: Path) -> None:
        """
        Download a single file with retry logic.

        Args:
            url: URL to download from
            output_path: Local path to save file

        Raises:
            requests.RequestException: If download fails after all retries
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                # Get file size for progress bar
                total_size = int(response.headers.get("content-length", 0))

                with open(output_path, "wb") as f:
                    if total_size > 0:
                        # Show progress for large files
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    else:
                        f.write(response.content)

                return  # Success

            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Download attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts")
                    raise

    def _process_ctx_isis3(self, img_path: Path) -> Optional[Path]:
        """
        Process CTX IMG through ISIS3 pipeline for geometric correction.

        This pipeline performs:
        1. mroctx2isis - Import to ISIS cube format
        2. spiceinit - Add SPICE geometry data
        3. ctxcal - Radiometric calibration (optional)
        4. cam2map - Map-project to uniform grid (fixes elliptical craters)
        5. isis2std - Export to GeoTIFF

        Args:
            img_path: Path to CTX .IMG file

        Returns:
            Path to georectified .tif file, or None if processing failed
        """
        product_id = img_path.stem
        tif_path = img_path.with_suffix(".tif")

        # Skip if already processed
        if tif_path.exists():
            logger.debug(f"Processed GeoTIFF already exists: {tif_path.name}")
            return tif_path

        # ISIS3 intermediate files
        isis_cube = img_path.parent / f"{product_id}.cub"
        cal_cube = img_path.parent / f"{product_id}.cal.cub"
        map_cube = img_path.parent / f"{product_id}.map.cub"

        try:
            # Step 1: Import CTX IMG to ISIS cube format
            logger.info(f"[1/5] mroctx2isis: {img_path.name}")
            subprocess.run(
                ["mroctx2isis", f"from={img_path}", f"to={isis_cube}"],
                check=True,
                capture_output=True,
                text=True,
            )

            # Step 2: Add SPICE geometry (spacecraft position/pointing)
            logger.info(f"[2/5] spiceinit: {isis_cube.name}")
            subprocess.run(
                ["spiceinit", f"from={isis_cube}"],
                check=True,
                capture_output=True,
                text=True,
            )

            # Step 3: Radiometric calibration (optional)
            working_cube = isis_cube
            if self.apply_calibration:
                logger.info(f"[3/5] ctxcal: {isis_cube.name}")
                subprocess.run(
                    ["ctxcal", f"from={isis_cube}", f"to={cal_cube}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                working_cube = cal_cube
            else:
                logger.info(f"[3/5] Skipping calibration (apply_calibration=False)")

            # Step 4: Map-project to uniform coordinate system
            # This corrects for spacecraft geometry and makes craters circular
            logger.info(f"[4/5] cam2map: {working_cube.name}")
            cam2map_args = [
                "cam2map",
                f"from={working_cube}",
                f"to={map_cube}",
                f"map=$ISISROOT/appdata/templates/maps/{self.map_projection}.map",
            ]

            # Set custom resolution if specified
            if self.map_resolution:
                cam2map_args.append(f"pixres=mpp")  # meters per pixel
                cam2map_args.append(f"resolution={self.map_resolution}")

            subprocess.run(
                cam2map_args,
                check=True,
                capture_output=True,
                text=True,
            )

            # Step 5: Export to GeoTIFF
            logger.info(f"[5/5] isis2std: {map_cube.name} -> {tif_path.name}")
            subprocess.run(
                [
                    "isis2std",
                    f"from={map_cube}",
                    f"to={tif_path}",
                    "format=GTiff",
                    "bittype=real",  # Preserve full dynamic range
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(f"Successfully processed {img_path.name} through ISIS3 pipeline")

            # Clean up intermediate files
            for temp_file in [isis_cube, cal_cube, map_cube, img_path]:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Removed intermediate file: {temp_file.name}")

            return tif_path

        except subprocess.CalledProcessError as e:
            logger.error(f"ISIS3 processing failed for {img_path.name}: {e.stderr}")
            # Clean up partial files
            for temp_file in [isis_cube, cal_cube, map_cube, tif_path]:
                if temp_file.exists():
                    temp_file.unlink()
            return None

        except FileNotFoundError as e:
            logger.error(
                f"ISIS3 command not found: {e}. Ensure ISIS3 is installed and $ISISROOT is set. "
                "See: https://isis.astrogeology.usgs.gov/documents/InstallGuide/"
            )
            return None

    def _convert_img_to_tif_gdal(self, img_path: Path) -> Optional[Path]:
        """
        Convert PDS IMG file to GeoTIFF using GDAL (simple format conversion).

        WARNING: This does NOT correct for spacecraft geometry. Features like
        craters may appear elliptical. Use ISIS3 pipeline for proper correction.

        Args:
            img_path: Path to .IMG file

        Returns:
            Path to converted .tif file, or None if conversion failed
        """
        tif_path = img_path.with_suffix(".tif")

        # Skip if already converted
        if tif_path.exists():
            logger.debug(f"GeoTIFF already exists: {tif_path.name}")
            return tif_path

        try:
            # Use gdal_translate to convert IMG to GeoTIFF
            # -of GTiff: Output format GeoTIFF
            # -co COMPRESS=LZW: Use LZW compression to reduce file size
            # -co TILED=YES: Create tiled TIFF for better performance
            # -co BIGTIFF=IF_SAFER: Use BigTIFF for large files
            cmd = [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "COMPRESS=LZW",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=IF_SAFER",
                str(img_path),
                str(tif_path),
            ]

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            logger.info(f"Converted {img_path.name} to GeoTIFF (GDAL - no geometry correction)")

            # Remove original IMG file to save space (keep XML label for metadata)
            img_path.unlink()
            logger.debug(f"Removed original IMG file: {img_path.name}")

            return tif_path

        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to convert {img_path.name} to GeoTIFF: {e.stderr}")
            return None
        except FileNotFoundError:
            logger.warning(
                "gdal_translate not found. Install GDAL to enable IMG->GeoTIFF conversion. "
                "Keeping IMG format."
            )
            return None

    def _load_manifest(self) -> Dict:
        """
        Load download manifest from disk.

        Returns:
            Manifest dictionary with downloaded_images tracking
        """
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Corrupt manifest file, starting fresh")

        return {
            "downloaded_images": {},
            "total_images": 0,
            "total_size_mb": 0.0,
            "last_updated": None,
        }

    def _save_manifest(self) -> None:
        """Save download manifest to disk."""
        self.manifest["last_updated"] = datetime.now().isoformat()

        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        logger.debug(f"Manifest saved to {self.manifest_path}")

    def _add_to_manifest(self, image_meta: Dict, img_path: Path) -> None:
        """
        Add downloaded image to manifest.

        Args:
            image_meta: Product metadata dict
            img_path: Path to downloaded IMG file
        """
        product_id = image_meta["product_id"]

        # Get file size
        file_size_mb = img_path.stat().st_size / (1024 * 1024)

        self.manifest["downloaded_images"][product_id] = {
            "product_id": product_id,
            "img_path": str(img_path),
            "label_path": str(img_path.with_suffix(".xml")),
            "center_lon": image_meta.get("center_lon"),
            "center_lat": image_meta.get("center_lat"),
            "emission_angle": image_meta.get("emission_angle"),
            "incidence_angle": image_meta.get("incidence_angle"),
            "solar_longitude": image_meta.get("solar_longitude"),
            "download_date": datetime.now().isoformat(),
            "file_size_mb": round(file_size_mb, 2),
        }

        self.manifest["total_images"] = len(self.manifest["downloaded_images"])
        self.manifest["total_size_mb"] = round(
            sum(img["file_size_mb"] for img in self.manifest["downloaded_images"].values()),
            2,
        )

    def get_manifest(self) -> Dict:
        """
        Get the current download manifest.

        Returns:
            Manifest dictionary with download statistics

        Example:
            >>> manifest = downloader.get_manifest()
            >>> print(f"Downloaded {manifest['total_images']} images")
            >>> print(f"Total size: {manifest['total_size_mb']:.1f} MB")
        """
        return self.manifest.copy()
