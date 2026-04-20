
# MarketMind AI

**Year:** 2026

## Authors

| Name | Affiliation | Contact |
|---|---|---|
| George David Tsitlauri | Dept. of Informatics & Telecommunications, University of Thessaly, Greece | gdtsitlauri@gmail.com |
| Vasileios Prodromos Tsaousis | Dept. of Economics, University of Cyprus, Cyprus | vptsaousis@gmail.com |

**Modern Behavioral Finance & Internet Psychology Analysis for Cryptocurrencies**

---

## Description
MarketMind AI is a complete, modular pipeline for analyzing the relationship between internet sentiment (behavioral biases like FOMO/FUD) and cryptocurrency prices, using advanced econometric and AI techniques.

## Pipeline Structure
1. **Data Collection**
   - Scraping Reddit (r/CryptoCurrency) & financial data (Yahoo Finance)
2. **Cleaning & Preprocessing**
   - Text cleaning, bot/spam removal
3. **AI Sentiment Analysis**
   - FinBERT (or similar) with CUDA (GTX 1650)
   - Daily aggregation: Positive/Negative/Fear-Greed
4. **Merging & Export**
   - Final CSV/Excel with all features
5. **Econometric Analysis**
   - OLS, VAR, Granger, VIF, ADF, ML (Random Forest)
   - Robustness checks per crypto/period
   - Automatic interpretation of results
6. **Reports & Visualizations**
   - Professional reports, plots, data dictionary

## Usage Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python scripts/merge_data.py
   python scripts/final_merge.py
   python scripts/run_regression.py
   ```
3. All results will be found in the `results/` folder.

## System Requirements
- Windows 10/11, Python 3.10+
- NVIDIA GTX 1650 (4GB VRAM) or better (for CUDA)
- 16GB RAM

## Data Dictionary
See `results/data_dictionary.txt` for a full description of all variables.

## Example Outputs
- Detailed reports: `results/Market_Analysis_Report*.txt`
- Plots: `results/*.png`

## Authors
- George David Tsitlauri
- Vasilis Prodromos Tsaousis

---
*For scientific use, presentation, or extension, contact the project authors.*

## Citation

```bibtex
@misc{tsitlauri2026marketmindai,
  author = {George David Tsitlauri},
  title  = {MarketMind AI: Behavioral Finance and Internet Psychology Analysis for Cryptocurrencies},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```
