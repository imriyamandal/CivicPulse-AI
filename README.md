# CivicPulse AI — Urban Problem Intelligence Engine

An AI-powered urban intelligence system that analyzes NYC 311 citizen service requests using data analytics, NLP, geospatial hotspot detection and machine learning — presented through a premium interactive dashboard.

> Complaints → Patterns → Locations → Risk/Concern Signals → Predictions → Explainable Insights

## What's in this folder

```
CivicPulse-AI-Final/
├── dashboard/
│   └── index.html              ← PREMIUM FRONTEND. Open this in any browser.
├── data/
│   └── civicpulse_dataset.csv  ← 500,000-row NYC 311 working sample (Jan 1–Mar 21, 2020)
├── models/
│   ├── nlp_model.joblib        ← trained TF-IDF + Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib
│   └── forecast_model.joblib   ← trained Random Forest daily-volume regressor
├── outputs/
│   ├── final_metrics.csv       ← accuracy, F1, MAE, RMSE, R², cluster counts
│   ├── urban_problem_scores.csv← Urban Problem Score (UPS) per borough
│   ├── hotspot_summary.csv     ← DBSCAN cluster centroids and sizes
│   ├── prediction_results.csv  ← actual vs. predicted daily volume (holdout)
│   ├── figures/                ← all notebook chart exports (PNG)
│   └── maps/civicpulse_hotspots.html ← Folium hotspot map
├── notebooks/
│   └── RiyaMandal_CivicPulseAI.ipynb ← full analysis notebook (source of truth)
├── docs/
│   └── Riya_CivicPulseAI_ProjectReport.docx
├── app.py                      ← original Streamlit reference dashboard
└── requirements.txt
```

## The premium frontend — `dashboard/index.html`

This is the "Bloomberg-style civic intelligence dashboard" described in the project brief, built as a single self-contained HTML file. **Double-click it, or open it in any browser — no server, no install, no dependencies beyond an internet connection for fonts/Chart.js.**

It is not a mockup: every number, chart and table is generated from the real dataset and the real trained models in this folder.

| Section | What it shows |
|---|---|
| **Overview** | KPI board (total requests, date window, boroughs, NLP accuracy), a live-data hero map of all 500K requests shaped like NYC, top categories, daily volume trend |
| **Problem Intelligence** | Top complaint categories, agency workload, common descriptors, category mix by borough |
| **NLP Classifier** | A **live, in-browser** re-implementation of your actual trained TF-IDF + Logistic Regression model — the real vocabulary (308 terms), IDF weights and coefficients were extracted from `nlp_model.joblib` / `tfidf_vectorizer.joblib` and run client-side in JavaScript. Type any complaint description and get the same prediction your Python model would give. |
| **Hotspot Map** | Every DBSCAN cluster from `hotspot_summary.csv`, plotted at true latitude/longitude with proportional sizing and borough coloring |
| **Urban Problem Score** | UPS = 0.40·volume + 0.25·growth + 0.35·resolution delay, per borough, with a click-through "why this area is flagged" profile |
| **Trends** | Monthly, day-of-week and hour-of-day patterns |
| **Forecast** | Actual-vs-predicted chart from `prediction_results.csv`, plus MAE/RMSE/R² from `final_metrics.csv` |
| **AI Insights** | Template-generated, fully grounded insight cards — every number is pulled from the tables above, nothing is invented |
| **Data Explorer** | Searchable, filterable, paginated table over a 1,500-record sample, with CSV export |
| **Methodology** | Pipeline diagram, dataset description, model summary, tech stack, scope & limitations |

Toggle light/dark mode top-right; the layout is fully responsive down to mobile.

## Running the original Streamlit app (optional)

The original reference dashboard (`app.py`) still works if you want the Python/Plotly version side by side:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regenerating the data / models

Everything in `data/`, `models/` and `outputs/` was produced by `notebooks/RiyaMandal_CivicPulseAI.ipynb`. Re-run the notebook top-to-bottom to reproduce or refresh every number the dashboard displays.

## Scope & limitations

- Reporting volume reflects citizen contact with 311, not verified defect severity.
- The Urban Problem Score is a project-defined comparison index, **not** an official city rating.
- The analysis window (Jan 1 – Mar 21, 2020) is a fixed, reproducible extraction, not a live feed — the real NYC 311 dataset updates continuously.
- Patterns shown are descriptive; they do not establish causality.
- This is an academic/portfolio project (AICTE · IBM SkillsBuild 2026 · BharatCares). No live deployment or official government endorsement is claimed.
