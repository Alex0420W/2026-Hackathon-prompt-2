# HarvestSight

**Geospatial-AI corn yield forecasting for the U.S. Corn Belt.**
A team project fine-tuning the NASA × IBM **Prithvi-EO-2.0-600M-TL** foundation model + multi-modal weather/soil/drought fusion → state-level yield forecasts for the **2025** growing season with full uncertainty cones.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prithvi](https://img.shields.io/badge/Foundation%20Model-Prithvi--EO--2.0--600M-0084FF)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)
[![NASA HLS](https://img.shields.io/badge/Imagery-HLS%20L30%2FS30-blue)](https://lpdaac.usgs.gov/products/hlsl30v002/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> CSU Geospatial AI Hackathon 2026 · Prompt 2 · Five states (IA / NE / WI / MO / CO) · Four in-season checkpoints (Aug 1 / Sep 1 / Oct 1 / End-of-Season).

---

## Why this project

> *"Will Iowa hit a record this year? How wide is the cone of plausible outcomes? Can we know in August?"*

Every August, USDA NASS publishes the first **WASDE** corn yield forecast — a survey-driven number that markets, traders, and policymakers rebuild their models around. WASDE state-level error sits around **5–10 bu/acre**.

HarvestSight is our team's alternative: an end-to-end **satellite-first** pipeline that reproduces (and matches) the WASDE state-level signal directly from raw NASA imagery + weather + drought + soil, **two months before the harvest concludes**, with calibrated uncertainty bands.

| Metric | HarvestSight 2024 (out-of-sample) | USDA WASDE | Pure NASS trend |
|---|---|---|---|
| **State-level mean abs error** | **8.1 bu/ac** | 5–10 bu/ac | 10–15 bu/ac |
| **County-level RMSE** | 16–22 bu/ac | n/a | n/a |
| **Calibration window** | Aug 1 onward | Aug WASDE | Annual |

> *Hindcast 2024 errors are genuine out-of-sample (2022 / 2023 used to fit the global linear calibration; 2024 was held out). State-level mean abs error on 2025 hindcast vs. early NASS estimates: **5.8 bu/ac**.*

---

## The forecast — 2025 season

Headline state-level point forecasts (bu/acre) and progression of the uncertainty cone as the season unfolds:

| State | Aug 1 | Sep 1 | Oct 1 | End-of-Season | vs 2024 actual |
|---|---|---|---|---|---|
| **Iowa**     | 218.4 [205.1 – 232.8] | 218.4 [209.8 – 227.8] | 218.4 [213.1 – 224.1] | 218.4 [215.7 – 221.3] | **+6.4** *(record territory)* |
| **Nebraska** | 190.5 [162.5 – 212.2] | 190.5 [172.3 – 204.6] | 190.5 [179.3 – 199.2] | 190.5 [184.9 – 194.9] | -2.5 |
| **Wisconsin**| 183.2 [169.5 – 196.6] | 183.2 [174.3 – 191.9] | 183.2 [177.7 – 188.5] | 183.2 [180.4 – 185.9] | +3.2 |
| **Missouri** | 178.5 [151.5 – 202.6] | 178.5 [161.0 – 194.1] | 178.5 [167.7 – 188.1] | 178.5 [173.1 – 183.3] | -3.5 |
| **Colorado** | 127.2 [75.2 – 165.5]  | 127.2 [93.4 – 152.1]  | 127.2 [106.4 – 142.5] | 127.2 [116.8 – 134.9] | +5.2 |

Brackets show the **p10–p90 uncertainty cone** (≈80% confidence interval). Cones narrow progressively from Aug→End-of-Season, mirroring USDA's own WASDE convergence pattern.

Full output: [`reports/forecasts/yield_with_uncertainty_2025.parquet`](reports/forecasts/yield_with_uncertainty_2025.parquet) · See [RESULTS.md](RESULTS.md) for the complete hindcast / calibration breakdown.

---

## Interactive dashboard

A production-style Streamlit app ships with the model — county choropleths, condition rollups, ensemble drilldown, and downloadable forecasts. Built by the team to make the model's outputs decision-ready for non-ML stakeholders:

```bash
streamlit run app.py
```

![Dashboard preview](Presentation%20Slides/Screenshot%202026-04-24%20193018.png)

---

## How it works

```
                ┌─────────────────────────────────────────────────────────┐
                │                  VISION BRANCH                          │
   HLS L30/S30 ─┤  Multi-temporal chips (T=3, 6 bands, 224×224 @ 30 m)    │
   (NASA, 2022 │           │                                              │
    – 2025)    │           ▼                                              │
                │  Prithvi-EO-2.0-600M-TL  ◄── LoRA r=16, qkv+proj        │
                │  (frozen backbone, mean-pool 768 patch tokens)          │
                │           │                                              │
                │           ▼  1280-dim embedding                         │
                └─────────┬─────────────────────────────────────────────┘
                          │
                ┌─────────┴─────────────────────────────────────────────┐
                │                  TABULAR BRANCH                       │
   gridMET     │  • Weather: GDD, precip, July heat days, July VPD    │
   USDM        │  • Drought: USDM dsci_mean / peak, D2+/D3+ pct       │
   gNATSGO     │  • Soil:    bulk density, CEC, clay, sand, OC, pH    │
                │           │                                            │
                │           ▼                                            │
                │  z-score norm → 64-dim MLP                            │
                └─────────┬─────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────────────┐
                │  Multi-modal fusion head │  →  yield (bu/acre)
                │  (1280 + 64) → MLP       │
                └──────────┬───────────────┘
                           │
              ┌────────────┴────────────────┐
              ▼                              ▼
   Ensemble (v4 + v4_restart)     Analog-year cone of uncertainty
              │                    k-NN over standardized weather +
              ▼                    USDM + NDVI features → empirical
   Calibration:                    p10/p25/p50/p75/p90 of historical
   ŷ = w·model + w'·prior_NASS    NASS deviations, re-anchored to
       + bias + per-state offset   the model's point forecast
              │
              ▼
         Final forecast + cone  →  state × checkpoint Parquet
```

### Why we built it this way

- **Foundation model + LoRA, not from scratch.** 4.7k HLS-era county-years would obliterate a 600M-parameter ViT trained from scratch. We use LoRA (r=16) to update ~0.3% of params (~2 M trainable) on `qkv` / `proj` — generalization-friendly, trains in ~4 h per fold on a single DGX Spark.
- **Multi-modal fusion, not satellite-only.** Pure imagery models cap out at ~16–18 bu/ac county-level RMSE on published benchmarks. We add a **tabular branch** (weather + soil + drought) to recover meteorological signal that's already latent in NDVI but noisier — a 64-dim MLP that fuses with the Prithvi embedding before the regression head.
- **County-level training, state-level reporting.** State-level training has only 100 samples (5 states × 20 years). County-level gives us ~9.4k rows. We aggregate to state at inference via CDL-acreage-weighted means.
- **Calibration on prior-year NASS.** Corn yields are 0.6–0.8 autocorrelated year-over-year. We fit a linear blend `ŷ = 0.805·model + 0.484·prior_yr − 47.6 + state_offset` to anchor the model against unrealistic excursions and absorb systematic per-state bias.
- **Analog-year uncertainty.** Our k-NN (k=5) over standardized weather + drought + NDVI retrieves the closest historical seasons; the **empirical** distribution of those years' actual NASS deviations becomes the cone — re-anchored, not re-modeled. This is the same logic climatologists use internally; we made it data-driven.

---

## Data sources

| Source | Use | Years | Access |
|---|---|---|---|
| [Prithvi-EO-2.0-600M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) | Vision backbone | – | Hugging Face |
| [HLS L30 / S30 v2.0](https://lpdaac.usgs.gov/products/hlsl30v002/) | 6-band Prithvi input | 2015–2025 | GEE `NASA/HLS/HLSL30/v002`, `HLSS30/v002` |
| [USDA CDL](https://nassgeodata.gmu.edu/CropScape/) | Corn pixel mask | 2008–2024 | GEE `USDA/NASS/CDL` |
| [USDA NASS QuickStats](https://quickstats.nass.usda.gov/api) | Yield labels + crop progress | 2005–2025 | REST API |
| [gridMET](https://www.climatologylab.org/gridmet.html) | Weather features | 2005–2025 | GEE `IDAHO_EPSCOR/GRIDMET` |
| [Landsat C2 L2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) | Pre-2015 NDVI backfill (analog only) | 2005–2014 | GEE `LANDSAT/{LT05,LE07,LC08}/C02/T1_L2` |
| [US Drought Monitor](https://droughtmonitor.unl.edu/DmData/GISData.aspx) | Drought severity (DSCI) | 2005–2025 | USDM CountyStatistics REST |
| [gNATSGO](https://www.nrcs.usda.gov/resources/data-and-reports/gridded-national-soil-survey-geographic-database-gnatsgo) | Soil features (AWC, OM, clay) | static | GEE `projects/sat-io/open-datasets/CSRL_soil_properties/*` |

**Volume:** ~1,920 HLS scenes / 486 GB / 3,592 multi-temporal training chips (Aug-1 / Sep-1 / Oct-1 each 224×224×6×3) · 6,560 NASS county-year labels · 2,344 trainable joins.

---

## Quick start

### Prerequisites

- Python 3.11+, CUDA-capable GPU (or MPS/CPU for inference-only)
- A Google Earth Engine account ([signup](https://earthengine.google.com/))
- A free [USDA NASS QuickStats API key](https://quickstats.nass.usda.gov/api)
- A Hugging Face token with access to [Prithvi-EO-2.0-600M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)

### Setup

```bash
git clone https://github.com/<you>/HarvestSight.git
cd HarvestSight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in NASS_API_KEY, EE_PROJECT, HF_TOKEN
earthengine authenticate
huggingface-cli login
```

### Run the dashboard (no training required)

```bash
streamlit run app.py        # opens at http://localhost:8501
```

### Reproduce the full pipeline

```bash
# 1. Download everything (idempotent — resumes from any partial state)
python scripts/download_all.py

# 2. Build labels + features
python scripts/training/build_labels_metadata.py
python scripts/training/build_features.py

# 3. Train the ensemble (sequential, ~4 h each on DGX Spark / A100)
python scripts/training/train_v4.py
python scripts/training/train_v4_restart.py

# 4. Inference + calibration + uncertainty cone
python scripts/inference/run_ensemble_final.py
python scripts/training/analog_year_uncertainty.py
```

`scripts/download_all.py` is the orchestrator — idempotent, validates env vars + GEE auth up front, reports per-step wall time, and supports `--skip`, `--only`, and `--dry-run`.

---

## Models trained

| Model | Config | Best Val RMSE | Notes |
|---|---|---|---|
| v1 | Frozen backbone, single checkpoint | 33.8 bu/ac | Baseline |
| v2 | 8 unfrozen blocks, full data | killed | Compute-infeasible (>17 h projected) |
| v3 / v3b | 8 unfrozen blocks, May-padded chips | 20.5 / 24.1 | Dropped — distribution shift at inference |
| **v4** | 4 unfrozen blocks, multi-checkpoint chips | **24.8 bu/ac** | Production |
| **v4_restart** | Warm restart from v4, fresh cosine cycle | **21.5 bu/ac** | Production (best) |

Final ensemble = `v4 + v4_restart`. Training was run by the team on a single **DGX Spark** node (Blackwell, 128 GB unified memory, BF16).

---

## Project layout

```
configs/                    project.yaml, terratorch_lora.yaml
scripts/
  download_all.py           idempotent orchestrator over every fetch step
  data/                     fetch_*.py, export_*.py — one script per source
  training/                 dataset.py, build_labels_metadata.py, train_v*.py,
                            analog_year_uncertainty.py
  inference/                run_ensemble_final.py, predict_2025.py, bias_correct.py
data/
  raw/                      one subdir per source (nass/, cdl/, hls/, ...)
  processed/
    chips/{state}/{fips}/{year}/{checkpoint}.zarr
    labels/                 county_yield.parquet, chip_metadata.parquet
    features/               state_checkpoint_features.parquet
models/
  checkpoints/              PyTorch Lightning checkpoints
  lora_adapters/            exported LoRA weights for re-use
reports/
  forecasts/                yield_with_uncertainty_2025.parquet
  figures/                  presentation plots
app.py                      Streamlit dashboard
RESULTS.md                  detailed hindcast / calibration writeup
```

---

## Limitations & honest caveats

- **2025 imagery is May-only.** No post-May 2025 HLS scenes were available at training time, so the 2025 point forecast is constant across all four checkpoints — only the cone narrows. The model's tabular branch still picks up post-May weather signal.
- **Hindcast 2022 / 2023 errors are inflated low** (those years anchored the global calibration). 2024 (8.1 bu/ac state mean abs error) is the genuine out-of-sample number.
- **Record-year extrapolation is bounded.** The model under-predicts MO 2024 by 22 bu/ac because 182 bu/ac is outside the training distribution for Missouri.
- **CDL 2024 used as the 2025 corn mask.** The official 2025 CDL is released after harvest; year-over-year corn pixel agreement is ~85%.
- **Uncertainty cone is empirical, not Bayesian.** It reflects historical year-to-year variability of analog seasons, not posterior parameter uncertainty.

---

## Team

Built collaboratively for the **CSU Geospatial AI Hackathon 2026**. Contributions span data engineering, foundation-model fine-tuning, calibration design, uncertainty quantification, and the Streamlit dashboard.

- **Alex Woods**
- **Hayley Smith**
- **Blaise Horsfall**
- **Kian Jiang**

## Acknowledgements

- **NASA + IBM** for releasing [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) under a permissive license — this project simply isn't possible without an open foundation model for Earth observation.
- **NASA HLS team** for the harmonized 30 m, 2–4 day surface reflectance product.
- **USDA NASS** for the QuickStats API and the multi-decade county-year yield record.
- **Climatology Lab** (UC Merced) for gridMET.
- **U.S. Drought Monitor** (UNL / NOAA / USDA) for the DSCI feed.
- **Colorado State University** for hosting the 2026 Geospatial AI Hackathon.

## License

[MIT](LICENSE) — fork it, ship it, improve it.
