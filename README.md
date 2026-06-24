# Stationarity-Aware Retrieval-Augmented Time Series Forecasting (KDD'26)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04135-b31b1b.svg)](https://arxiv.org/abs/2606.04135)

This repository provides the official implementation of [**SARAF**](https://arxiv.org/abs/2606.04135), accepted by KDD 2026. 

## What is SARAF?

Retrieval-augmented forecasting usually assumes that similar historical patterns lead to similar future trajectories. However, this assumption can be unreliable for real-world time series, where different datasets exhibit different levels of stationarity. In highly non-stationary series, such as exchange-rate-like data, two historical windows may look similar in the past but evolve very differently in the future.

**SARAF** addresses this issue by making retrieval stationarity-aware. Instead of relying only on temporal similarity, SARAF adaptively combines:

- **Time-aligned retrieval**, which strengthens temporally meaningful historical evidence;
- **Diversity-aware retrieval**, which avoids redundant neighbors and covers heterogeneous historical regimes;
- **Stationarity-aware aggregation**, which controls how retrieved futures are fused according to the stationarity of the dataset.

In short, SARAF asks not only:

> **“Which past segments look similar to the query?”**

but also:

> **“When can their future trajectories be trusted?”**

This makes retrieval-augmented forecasting more robust under non-stationary settings while preserving the benefits of similarity-based retrieval on more stable datasets.

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
# ETTh + ETTm (seq_len=720)
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

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@misc{zhou2026saraf,
  title         = {Stationarity-Aware Retrieval-Augmented Time Series Forecasting},
  author        = {Zhou, Shiqiao and Sch{\"o}ner, Holger and Wu, Zipeng and Fouch{\'e}, Edouard and Wilson, IAG and Wang, Shuo},
  year          = {2026},
  doi           = {10.48550/arXiv.2606.04135},
  url           = {https://arxiv.org/abs/2606.04135}
}
