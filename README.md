# Integrating News Sentiment and Global Event Data into Short-Term Stock Price Forecasting

**An Explainable Machine Learning Approach for the FTSE 100**

MSc Applied Data Science Dissertation — University of Essex, 2025

---

## Overview

This project investigates whether incorporating news sentiment and macroeconomic event data can improve short-term directional forecasting of the FTSE 100 index. It builds an end-to-end pipeline that:

1. **Collects and cleans** FTSE 100 price data, 53,330 financial news headlines (CNBC, Guardian, Reuters), and a UK macroeconomic event calendar
2. **Extracts sentiment** from headlines using [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone), a transformer model fine-tuned for financial text
3. **Engineers 37 features** spanning technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR), sentiment aggregates (daily average, 5-day momentum, 10-day volatility), and a binary event-day flag
4. **Trains and compares** Random Forest and XGBoost classifiers on a chronological 80/20 split (403 trading days, Mar 2018 – Jul 2020)
5. **Interprets predictions** using SHAP (global feature importance) and LIME (local prediction explanations)
6. **Backtests** a simple long-only trading strategy against a buy-and-hold benchmark

## Key Results

| Metric | XGBoost (Hybrid) | Random Forest | Buy-and-Hold |
|---|---|---|---|
| Accuracy | **65.4%** | 49.4% | — |
| F1-Score | **0.714** | 0.468 | — |
| ROC-AUC | **0.711** | 0.588 | — |
| Total Return | **36.19%** | — | 22.14% |
| Sharpe Ratio | **4.43** | — | 2.35 |
| Max Drawdown | −7.54% | — | −6.70% |

**Key findings:**
- Technical indicators (Bollinger Bands, lagged returns, RSI) were the dominant predictive features
- News sentiment acted as a **conditional modifier** — amplifying or dampening technical signals rather than serving as a standalone trigger
- Scheduled macroeconomic events showed **no significant directional impact** at the daily horizon (p = 0.506), consistent with semi-strong market efficiency
- SHAP provided stable global explanations suitable for model governance; LIME offered intuitive local explanations useful for individual trade signal diagnosis

## Repository Structure

```
├── notebooks/
│   └── FTSE_100_News_Sentiment___Event_Analysis.ipynb   # Full analysis pipeline
├── data/
│   ├── CNBC_Headlines.csv              # CNBC financial news headlines
│   ├── Guardian_Headlines.csv          # Guardian news headlines
│   ├── Reuters_Headlines.csv           # Reuters financial news headlines
│   ├── FTSE_100_Prices_2019-2024.csv   # FTSE 100 daily OHLCV data
│   ├── Event_Calendar.csv             # UK macroeconomic event calendar
│   └── Daily_News_2019-2024.csv       # Supplementary daily news data
├── docs/
│   └── (dissertation PDF available on request)
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

```
Price Data ──→ Returns + Technical Indicators ─┐
                                                ├──→ Feature Matrix (37 features) ──→ XGBoost ──→ Predictions ──→ Backtest
News Headlines ──→ FinBERT Sentiment Scores ───┤                                        │
                                                │                                    SHAP + LIME
Event Calendar ──→ Binary Event-Day Flag ───────┘                                   (Explainability)
```

### Feature Engineering
- **Technical:** SMA (5, 20-day), RSI, MACD, Bollinger Bands, ATR, rolling volatility, lagged returns (1, 2, 3, 5-day), volume changes
- **Sentiment:** Daily average FinBERT score, 5-day sentiment momentum, 10-day sentiment volatility
- **Event:** Binary flag for scheduled UK macroeconomic events (BoE decisions, ONS releases)

### Models
- **Random Forest** (baseline): 100 trees, default hyperparameters
- **XGBoost** (enhanced): Tuned via randomised search with time-series cross-validation, binary logistic objective

### Explainability
- **SHAP:** Global feature importance via TreeExplainer — identifies which features the model relies on across all predictions
- **LIME:** Local explanations for individual predictions — translates model logic into human-readable decision rules

## Setup & Usage

### Prerequisites
- Python 3.8+
- ~4 GB RAM (FinBERT inference)

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/ftse-sentiment-forecasting.git
cd ftse-sentiment-forecasting
pip install -r requirements.txt
```

### Running the Analysis

Open the notebook in Jupyter or Google Colab:

```bash
jupyter notebook notebooks/FTSE_100_News_Sentiment___Event_Analysis.ipynb
```

The notebook is self-contained and runs end-to-end: data loading → preprocessing → EDA → feature engineering → modelling → XAI → backtesting.

**Note:** FinBERT inference on the full 53K headline corpus is computationally expensive on CPU. The notebook samples 1,000 headlines as a proof-of-concept. For full corpus scoring, a GPU runtime (e.g., Google Colab Pro) is recommended.

## Limitations

- **Regime-specific:** Trained/tested during COVID-19 volatility (2018–2020); performance may differ in other market conditions
- **Simplified backtest:** Ignores transaction costs, slippage, and market impact — reported returns are an upper bound
- **Partial sentiment coverage:** FinBERT applied to 1,000 sampled headlines, not the full 53K corpus
- **Coarse event feature:** Single binary flag; a multi-class event taxonomy could provide richer signal

## Future Work

- Score the full news corpus using GPU-accelerated FinBERT inference
- Implement entity/aspect-based sentiment analysis
- Build a multi-class event taxonomy (monetary policy, geopolitical, earnings, etc.)
- Test across different market regimes (pre-COVID, post-COVID)
- Add transaction costs and position sizing for realistic backtesting

## Tools & Technologies

Python, Pandas, NumPy, scikit-learn, XGBoost, Transformers (HuggingFace), FinBERT, SHAP, LIME, Plotly, Matplotlib, Seaborn, yfinance, ta (Technical Analysis)

## Author

**Huraira Qamar**
- MSc Applied Data Science — University of Essex
- [LinkedIn](https://www.linkedin.com/in/huraira-qamar)

## License

This project is for educational and portfolio purposes. The code and analysis are shared for reference; the datasets are sourced from publicly available resources.
