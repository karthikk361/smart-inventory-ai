# 📦 Smart Inventory & Demand Prediction System

> AI-powered demand forecasting, inventory optimization, and anomaly detection — built for scale.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 🚀 Features

| Module | Description |
|--------|-------------|
| **Demand Forecasting** | Random Forest model with 37 engineered features (lag, rolling stats, Fourier, EWMA) |
| **Inventory Status** | Real-time SKU-level health dashboard with reorder recommendations |
| **Anomaly Detection** | Z-score based spike detection (σ > 2.5) across all categories |
| **Model Insights** | Feature importances, residual analysis, actual vs predicted plots |
| **Live Predictor** | Real-time demand prediction with configurable inputs |

## 📊 Model Performance

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| **Random Forest** ⭐ | 28.5 | 66.5 | **0.857** | **18.6%** |
| Gradient Boosting | 30.5 | 74.4 | 0.821 | 20.6% |
| Ridge Baseline | 42.5 | 81.3 | 0.786 | 34.0% |

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Streamlit` · `Plotly`

## 📁 Project Structure

```
inventory_project/
├── app.py                    # Main Streamlit dashboard
├── requirements.txt          # Dependencies
├── data/
│   ├── generate_dataset.py   # Synthetic data generation
│   ├── sales_data.csv        # 54,800 daily sales records
│   └── product_catalog.csv   # 50 SKU catalog
├── train_model.py            # ML pipeline (feature eng + training)
└── models/
    ├── best_model.pkl        # Trained Random Forest
    ├── model_metrics.json    # Evaluation metrics
    ├── feature_importance.csv
    ├── inventory_status.csv
    ├── anomaly_data.csv
    └── processed_data.csv
```

## ⚡ Run Locally

```bash
git clone https://github.com/your-username/smart-inventory-ai
cd smart-inventory-ai/inventory_project
pip install -r requirements.txt
python data/generate_dataset.py
python train_model.py
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `inventory_project/app.py`
5. Click **Deploy** — live in ~2 minutes!

> **Note:** Run `generate_dataset.py` and `train_model.py` locally first, then commit the `models/` folder to GitHub before deploying (the `.pkl` and `.csv` files are needed at runtime).

---

**Built by Karts · Delhi Technological University**
