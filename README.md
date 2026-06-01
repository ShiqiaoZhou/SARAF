# Stationarity-Aware Retrieval-Augmented Time Series Forecasting (KDD'26)

This repository provides the official PyTorch implementation of the KDD 2026 paper, **Stationarity-Aware Retrieval-Augmented Time Series Forecasting** (SARAF). SARAF is a retrieval-augmented forecasting framework that adaptively balances retrieval relevance and diversity according to dataset stationarity.

<p align="center">
  <img src="fig/SARAF.png" alt="Overview of the SARAF framework" width="85%">
</p>

---

## Required Packages

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Create a `./data` directory and place dataset files inside:

```
mkdir -p ./data
```

> All standard benchmark datasets (ETT, Electricity, Exchange, Traffic, Solar) can be downloaded from the [Autoformer Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2).

---

## Usage

### Run with Scripts (Recommended)

We provide per-dataset bash scripts under `./scripts/`. Each script runs experiments across multiple prediction lengths and random seeds.

```bash
# ETTh1 (seq_len=720)
bash scripts/ETTh_720.sh

# Electricity
bash scripts/elec_720.sh

# Exchange Rate
bash scripts/exchange_rate_720.sh

# Traffic
bash scripts/traffic_720.sh

# Solar
bash scripts/solar_720.sh

```

## Acknowledgement

This code is based on [RAFT](https://arxiv.org/abs/2505.04163) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library). We thank the authors for their open-source contributions.
