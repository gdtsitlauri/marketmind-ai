# MarketMind AI

**Behavioral Finance and Internet Psychology Analysis for Cryptocurrency Markets**

**Authors:** George David Tsitlauri, Vasileios Prodromos Tsaousis  
**Contacts:** gdtsitlauri@gmail.com, vptsaousis@gmail.com  
**Website:** gdtsitlauri.dev  
**GitHub:** github.com/gdtsitlauri  
**Year:** 2026

MarketMind AI is a reproducible research pipeline that links online sentiment and behavioral-finance proxies to short-horizon cryptocurrency market dynamics. The repository combines Reddit sentiment extraction, market-data ingestion, feature engineering, econometric analysis, and report generation.

## Evidence Status

| Item | Current status |
| --- | --- |
| End-to-end scripted pipeline | Present |
| Real market and Reddit-derived inputs | Present |
| Econometric report artifacts | Present |
| Machine-learning benchmark | Present |
| Strong causal identification claims | Not supported by the current sample size |

## Research Positioning

The strongest evidence-backed story in the repository is:

> short-horizon cryptocurrency returns show measurable association with volatility and negative sentiment proxies, while the current data coverage is too limited for strong causal claims.

That is a credible and useful applied-finance project. It is better than overselling a small-sample result as definitive market science.

## Pipeline Overview

| Stage | Main script | Output |
| --- | --- | --- |
| Sentiment extraction | `scripts/reddit_kaggle_ai.py` | daily Reddit sentiment aggregates |
| Market data + features | `scripts/merge_data.py` | per-asset market features |
| Final analytical merge | `scripts/final_merge.py` | `exports/ULTIMATE_Data.csv` / `.xlsx` |
| Econometric analysis | `scripts/run_regression.py` | `results/Market_Analysis_Report.txt` and plots |

## Data Basis

- Reddit posts from `r/CryptoCurrency`
- Yahoo Finance cryptocurrency market data
- Fear and Greed index
- engineered volatility and technical indicators

The merged analytical dataset covers 332 rows, but the strict complete-case econometric specification used in the paper reduces the effective sample for the main OLS model to 20 observations. That sample-size reduction is a central limitation and is treated explicitly in the paper.

## Current Findings

Source: `results/Market_Analysis_Report.txt`

- OLS: `R^2 = 0.430`, adjusted `R^2 = 0.277`
- Volatility is the clearest statistically significant driver in the main specification
- Negative sentiment is directionally informative but not individually strong at conventional thresholds in the current sample
- Random Forest achieves stronger in-sample fit than OLS, suggesting non-linear effects

## Repository Layout

```text
scripts/
  reddit_kaggle_ai.py
  merge_data.py
  final_merge.py
  run_regression.py
exports/
results/
  Market_Analysis_Report.txt
  *.png
paper/
  MarketMindAI_paper.tex
requirements.txt
```

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/reddit_kaggle_ai.py
python scripts/merge_data.py
python scripts/final_merge.py
python scripts/run_regression.py
```

## Limitations

- Effective econometric sample size is small.
- Machine-learning performance is currently best interpreted as descriptive or in-sample.
- Sentiment signals come from a limited social source mix and time range.

## Citation

```bibtex
@misc{tsitlauri2026marketmindai,
  author = {George David Tsitlauri and Vasileios Prodromos Tsaousis},
  title  = {MarketMind AI: Behavioral Finance and Internet Psychology Analysis for Cryptocurrencies},
  year   = {2026},
  url    = {https://github.com/gdtsitlauri}
}
```
