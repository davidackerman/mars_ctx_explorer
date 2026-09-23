# Data

Everything here except this file is gitignored.

- `raw/ctx/` — locally downloaded CTX products for the local viewer and index builders
  (`pixi run ctx-pipeline`, `pixi run ctx-download`).
- `isis3data/` — ISIS3 SPICE kernels and calibration data, if you point `ISISDATA` here.

The Murray Lab index builders stream tiles and write nothing here. Built indexes, tile caches
and rebuild logs go to `outputs/`, also gitignored.
