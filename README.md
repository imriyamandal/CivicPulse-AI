# CivicPulse AI — Urban Problem Intelligence Engine

<p align="center">

### AI-Powered Urban Intelligence from Citizen Service Requests

**Complaints → Patterns → Locations → Risk Signals → Predictions → Explainable Insights**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analytics-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visualization-3F4F75?logo=plotly)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)

</p>

---

## Overview

**CivicPulse AI** is an AI-powered **Urban Problem Intelligence Engine** that transforms large-scale citizen service request data into structured, interpretable intelligence about urban problems.

The project analyzes **NYC 311 citizen service requests** using:

- Exploratory Data Analysis
- Natural Language Processing
- Text Classification
- Geospatial Analysis
- DBSCAN Hotspot Detection
- Urban Problem Scoring
- Time-Series Feature Engineering
- Machine Learning Forecasting
- Interactive Visualization
- Explainable, data-grounded insights

Instead of simply displaying complaint counts, CivicPulse AI follows an intelligence pipeline:

> **Citizen Reports → Data Cleaning → Patterns → Spatial Hotspots → Problem Signals → Forecasting → Explainable Insights**

The goal is to demonstrate how citizen-generated service-request data can be transformed into a structured analytical system for understanding **where problems occur, what problems dominate, how they vary across locations and time, and what future request volumes may look like.**

---

## Why CivicPulse AI?

Urban service-request datasets contain valuable signals about recurring problems in cities.

However, raw complaint records are difficult to interpret directly because they contain:

- Hundreds of thousands of records
- Multiple agencies and complaint categories
- Free-text descriptions
- Geographic coordinates
- Temporal patterns
- Different borough-level distributions
- Missing or inconsistent values

CivicPulse AI converts this raw information into a unified analytical workspace.

### The system answers questions such as:

- What are the most frequently reported urban problems?
- Which agencies receive the highest workload?
- Which complaint categories dominate different boroughs?
- What spatial areas contain concentrated complaint activity?
- What complaint descriptions can be classified automatically?
- Which boroughs show stronger problem signals according to the project's scoring framework?
- How does complaint volume change over time?
- Can historical request patterns help estimate future daily request volume?
- What evidence supports each generated insight?

---

## Core Capabilities

| Capability | Description |
|---|---|
| **Urban Problem Analytics** | Identifies dominant complaint categories, agencies, descriptors and borough-level patterns |
| **NLP Classification** | Classifies complaint descriptions using TF-IDF + Logistic Regression |
| **Geospatial Intelligence** | Detects spatial concentrations using DBSCAN |
| **Hotspot Detection** | Produces cluster centroids, sizes and geographic distributions |
| **Urban Problem Score** | Computes a project-defined comparative problem index by borough |
| **Temporal Intelligence** | Analyzes monthly, weekly and hourly request patterns |
| **Forecasting** | Predicts daily request volume using a Random Forest regression model |
| **Interactive Dashboard** | Provides a premium interface for exploring project outputs |
| **Data Explorer** | Enables searching, filtering and exporting a sample of records |
| **Explainable Insights** | Generates evidence-grounded insight cards using project outputs |
| **Methodology Workspace** | Documents the complete analytical pipeline and limitations |

---

## Project Architecture

```text
                         NYC 311 SERVICE REQUESTS
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Data Loading   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Data Cleaning &  │
                         │ Preprocessing    │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
       Exploratory             NLP Pipeline       Geospatial
       Data Analysis                                Analysis
              │                   │                   │
              ▼                   ▼                   ▼
       Pattern Analysis      TF-IDF Features      DBSCAN
       Category Analysis     Logistic Regression   Hotspots
       Temporal Analysis          │                   │
              │                    ▼                   │
              │              Classification           │
              │                                      │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Urban Problem Scoring  │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Forecasting Pipeline   │
                     │ Random Forest Regressor│
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Evidence-Grounded      │
                     │ Intelligence Layer     │
                     └────────────┬───────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Interactive Dashboard  │
                     └────────────────────────┘
```

---

## Dataset

### Source

The project is based on **NYC 311 citizen service request data**.

The working analysis uses a:

> **500,000-row NYC 311 working sample**

covering:

> **January 1, 2020 – March 21, 2020**

The dataset contains information used for analytical, NLP, temporal and spatial modeling.

Representative fields include:

- Unique request identifier
- Created date
- Agency
- Agency name
- Complaint type
- Descriptor
- Borough
- Status
- Latitude
- Longitude
- Resolution time

### Dataset Availability

The raw working dataset is intentionally **not versioned in this GitHub repository** because the CSV is approximately **318 MB**, exceeding GitHub's standard 100 MB file-size limit.

The local project structure expects:

```text
data/
└── civicpulse_dataset.csv
```

The dataset should therefore be treated as a **local/generated project artifact**, while the analysis notebook, models, outputs and application code remain version controlled.

> **Note:** The GitHub repository does **not** contain the 318 MB raw CSV. This keeps the repository lightweight while the analytical source code and generated analytical outputs remain fully available.

---

## Dataset Processing

The analytical workflow performs data preparation before modeling. Typical processing includes:

1. Loading the NYC 311 records
2. Parsing date/time fields
3. Handling missing values
4. Standardizing categorical values
5. Preparing complaint text
6. Preparing geographic coordinates
7. Calculating resolution-related features
8. Creating temporal features
9. Preparing modeling datasets
10. Generating analytical outputs

The complete implementation is documented in:

```text
notebooks/RiyaMandal_CivicPulseAI.ipynb
```

The notebook is treated as the project's **source of truth** for the analytical pipeline.

---

## Analytics Modules

### 1. Exploratory Data Analysis

The EDA layer establishes the basic structure and distribution of the dataset. It examines:

- Complaint categories
- Agency distribution
- Borough distribution
- Complaint status
- Resolution time
- Temporal trends
- Geographic distribution
- Category-level patterns

Generated visualizations are available under:

```text
outputs/figures/
```

### 2. NLP Classification

CivicPulse AI includes a Natural Language Processing pipeline for classifying complaint descriptions.

**Pipeline**

```text
Complaint Description
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Feature Representation
        │
        ▼
Logistic Regression
        │
        ▼
Predicted Complaint Class
```

**Technologies:** TF-IDF Vectorizer · Logistic Regression · Scikit-learn

The trained artifacts are stored as:

```text
models/
├── nlp_model.joblib
└── tfidf_vectorizer.joblib
```

**Browser-Based NLP Demonstration**

The premium frontend includes an in-browser implementation of the trained NLP classifier. The dashboard uses the extracted vocabulary, IDF weights, and model coefficients to reproduce the trained classification logic inside the browser — allowing users to enter a complaint description and observe the corresponding model prediction without requiring a Python inference request for the demonstration.

### 3. Geospatial Hotspot Detection

CivicPulse AI uses **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** to identify geographic concentrations of service requests.

**Pipeline**

```text
Latitude + Longitude
        │
        ▼
Geospatial Preprocessing
        │
        ▼
DBSCAN Clustering
        │
        ▼
Spatial Clusters
        │
        ▼
Cluster Centroids
        │
        ▼
Hotspot Intelligence
```

The resulting hotspot summary contains:

- Cluster identifier
- Primary complaint
- Primary borough
- Average latitude
- Average longitude
- Complaint count

Output: `outputs/hotspot_summary.csv`
Interactive map: `outputs/maps/civicpulse_hotspots.html`

### 4. Urban Problem Score

CivicPulse AI introduces a project-defined **Urban Problem Score (UPS)** to compare borough-level problem signals.

```text
UPS = 0.40 × Volume + 0.25 × Growth + 0.35 × Resolution Delay
```

The score combines request volume, growth signal, and resolution delay to create a comparative analytical index.

> **Important:** The Urban Problem Score is a project-defined analytical metric intended for comparative exploration. It is **not** an official NYC government rating and **not** a measure of verified physical defect severity.

Output: `outputs/urban_problem_scores.csv`

### 5. Temporal Intelligence

The project analyzes request patterns across multiple time dimensions:

- **Monthly analysis** — identifies changes in request volume across the analysis window
- **Day-of-week analysis** — examines differences in request activity from Monday to Sunday
- **Hour-of-day analysis** — examines intraday request patterns

These analyses help identify recurring temporal behavior in citizen service requests.

### 6. Forecasting

CivicPulse AI includes a machine learning forecasting pipeline for daily request volume.

**Model:** Random Forest Regressor

**Feature engineering includes temporal signals such as:**

- Lagged request volume
- Rolling statistics
- Calendar features
- Historical daily patterns

**Pipeline**

```text
Historical Daily Requests
          │
          ▼
Temporal Feature Engineering
          │
          ▼
Lag / Rolling Features
          │
          ▼
Time-Based Train/Test Split
          │
          ▼
Random Forest Regression
          │
          ▼
Daily Volume Prediction
```

Trained model: `models/forecast_model.joblib`
Prediction output: `outputs/prediction_results.csv`

The dashboard displays actual values, predicted values, and forecasting metrics.

---

## Explainability & Evidence Grounding

CivicPulse AI is designed so that dashboard insights are grounded in project-generated analytical outputs. The system does not present arbitrary numbers as analytical conclusions.

Insight cards are generated from:

```text
final_metrics.csv
hotspot_summary.csv
urban_problem_scores.csv
prediction_results.csv
```

**Intelligence principle**

```text
Raw Data → Validated Analytical Output → Structured Evidence → Insight
```

This makes the dashboard's analytical statements traceable to the project's underlying results.

---

## Premium Dashboard

CivicPulse AI provides a standalone premium frontend:

```text
dashboard/index.html
```

The interface is designed around a **civic intelligence / Bloomberg-style analytical workspace** rather than a conventional academic dashboard. It is responsive, interactive, data-driven, browser-based, designed for analytical exploration, and available in light and dark modes.

### Dashboard Sections

**Overview** — Total requests, analysis date window, borough coverage, NLP model performance, geographic visualization, top complaint categories, daily volume trends.

**Problem Intelligence** — Complaint categories, agency workload, common descriptors, borough-level category distributions.

**NLP Classifier** — Users enter complaint descriptions and observe the model's predicted class, powered by the trained model's extracted parameters running in-browser.

**Hotspot Map** — DBSCAN-derived spatial clusters shown by latitude, longitude, cluster size, borough, and complaint category.

**Urban Problem Score** — Borough-level UPS information with the underlying components contributing to the score.

**Trends** — Monthly trends, day-of-week patterns, hour-of-day patterns.

**Forecast** — Actual vs. predicted daily request volume, forecasting performance metrics, model output from the project's prediction pipeline.

**AI Insights** — Template-generated, evidence-grounded analytical observations summarizing project outputs rather than fabricating unsupported information.

**Data Explorer** — Search, filtering, pagination, record exploration, CSV export over a manageable record sample.

**Methodology** — Documents the analytical pipeline, dataset, NLP approach, clustering approach, forecasting model, Urban Problem Score, technology stack, scope, and limitations.

---

## Repository Structure

```text
CivicPulse-AI-Final/
│
├── dashboard/
│   └── index.html                     # Premium interactive frontend
│
├── data/
│   └── civicpulse_dataset.csv         # Local 500K-row working dataset (not in GitHub)
│
├── models/
│   ├── nlp_model.joblib               # TF-IDF + Logistic Regression classifier
│   ├── tfidf_vectorizer.joblib        # Trained TF-IDF vectorizer
│   └── forecast_model.joblib          # Random Forest forecasting model
│
├── outputs/
│   ├── final_metrics.csv
│   ├── urban_problem_scores.csv
│   ├── hotspot_summary.csv
│   ├── prediction_results.csv
│   │
│   ├── figures/
│   │   ├── agency_distribution.png
│   │   ├── borough_distribution.png
│   │   ├── complaint_categories.png
│   │   ├── day_of_week.png
│   │   ├── feature_importance.png
│   │   ├── hotspot_detection.png
│   │   ├── monthly_trend.png
│   │   ├── nlp_confusion_matrix.png
│   │   ├── prediction_results.png
│   │   ├── resolution_by_category.png
│   │   ├── resolution_distribution.png
│   │   ├── spatial_distribution.png
│   │   ├── status_distribution.png
│   │   └── urban_problem_score.png
│   │
│   └── maps/
│       └── civicpulse_hotspots.html
│
├── notebooks/
│   └── RiyaMandal_CivicPulseAI.ipynb  # Complete analytical workflow
│
├── docs/
│   └── Riya_CivicPulseAI_ProjectReport.docx
│
├── app.py                             # Streamlit reference dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technology Stack

**Programming & Analytics**

| Technology | Purpose |
|---|---|
| **Python** | Core development and analytics |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computation |
| **Scikit-learn** | NLP, classification, clustering and ML |
| **Joblib** | Model serialization |
| **Jupyter Notebook** | Reproducible analysis |

**Visualization**

| Technology | Purpose |
|---|---|
| **Plotly** | Interactive analytical charts |
| **Matplotlib** | Static analytical visualizations |
| **Folium** | Interactive geospatial mapping |
| **Chart.js** | Browser-based dashboard visualizations |

**Dashboard**

| Technology | Purpose |
|---|---|
| **HTML5** | Standalone frontend |
| **CSS3** | Responsive premium UI |
| **JavaScript** | Dashboard interactions |
| **Streamlit** | Python-based reference dashboard |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/imriyamandal/CivicPulse-AI.git
cd CivicPulse-AI
```

**2. Create a virtual environment**

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Premium Dashboard

The main frontend is:

```text
dashboard/index.html
```

It is a standalone browser-based dashboard. Simply open `dashboard/index.html` in a modern web browser.

> **Note:** Browser access to external libraries/fonts may require an internet connection depending on the frontend implementation.

---

## Running the Streamlit Dashboard

The repository also contains the original Python/Plotly dashboard.

```bash
streamlit run app.py
```

If the `streamlit` executable is not directly recognized on Windows, use:

```bash
python -m streamlit run app.py
```

The Streamlit application provides a Python-based analytical workspace with sections for: Overview · Data Explorer · Problem Analytics · Temporal Trends · NLP Classification · Hotspots · Urban Problem Score · Forecast · Methodology.

---

## Reproducing the Analysis

The complete analytical workflow is contained in:

```text
notebooks/RiyaMandal_CivicPulseAI.ipynb
```

The notebook is the project's **source of truth**. The workflow can be reproduced by executing the notebook from top to bottom after making the required dataset available locally.

The notebook produces the `data/`, `models/`, and `outputs/` directories, including:

- Trained NLP model
- TF-IDF vectorizer
- Forecasting model
- Analytical metrics
- Hotspot summaries
- Urban Problem Scores
- Forecast results
- Visualization exports
- Interactive hotspot map

---

## Generated Outputs

| Output | Description |
|---|---|
| `outputs/final_metrics.csv` | Project-level analytical and model metrics (NLP classification, forecasting, clustering) |
| `outputs/urban_problem_scores.csv` | Borough-level Urban Problem Score results |
| `outputs/hotspot_summary.csv` | DBSCAN cluster information |
| `outputs/prediction_results.csv` | Actual vs. predicted daily request volumes for the forecasting evaluation period |
| `outputs/figures/` | Exported analytical visualizations |
| `outputs/maps/civicpulse_hotspots.html` | Generated interactive hotspot visualization |

---

## Model Artifacts

```text
models/
├── nlp_model.joblib
├── tfidf_vectorizer.joblib
└── forecast_model.joblib
```

- **NLP:** TF-IDF → Logistic Regression
- **Forecasting:** Temporal Features → Random Forest Regressor
- **Spatial Intelligence:** Latitude + Longitude → DBSCAN

---

## Analytical Pipeline

```text
                 NYC 311 DATA
                      │
                      ▼
              Data Preprocessing
                      │
                      ▼
              Exploratory Analysis
                      │
        ┌─────────────┼──────────────┐
        │             │              │
        ▼             ▼              ▼
       NLP        Geospatial      Temporal
        │             │              │
        ▼             ▼              ▼
   TF-IDF +       DBSCAN         Feature
   Logistic       Hotspots       Engineering
   Regression         │              │
        │              │              ▼
        │              │        Random Forest
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
             Urban Problem Score
                       │
                       ▼
             Evidence-Grounded
                 Insights
                       │
                       ▼
             Interactive Dashboard
```

---

## Project Design Principles

1. **Data Grounding** — Analytical statements originate from actual project outputs.
2. **Reproducibility** — The complete analytical workflow is maintained in the Jupyter notebook.
3. **Explainability** — Model outputs and analytical scores are presented with supporting information rather than as unexplained black-box conclusions.
4. **Separation of Analysis and Presentation** — The notebook produces analytical artifacts; the dashboard consumes those artifacts for visualization and exploration.
5. **Responsible Interpretation** — Citizen service-request volume is treated as a signal of reported activity, not automatically as proof of physical problem severity.

---

## Scope & Limitations

CivicPulse AI is an analytical and academic/portfolio project.

1. **Citizen reports are not direct measurements of severity.** 311 request volume represents citizen contact with the service system. A higher number of reports does not necessarily mean a problem is objectively more severe.
2. **Urban Problem Score is project-defined.** It is a custom analytical index developed for this project and is **not an official NYC government rating or public-sector risk score.**
3. **Fixed historical analysis window.** The project analyzes a fixed working sample covering January 1, 2020 – March 21, 2020. It should not be interpreted as a live representation of current NYC conditions. The actual NYC 311 dataset is continuously updated independently of this project.
4. **Descriptive patterns do not establish causality.** Observed relationships between complaint volume, location, time, categories, and resolution behavior should not automatically be interpreted as causal relationships.
5. **Forecasting is not certainty.** Machine learning predictions are estimates derived from historical patterns. Actual future service-request volume may differ from model predictions.
6. **Dataset versioning.** The large raw dataset is not stored in GitHub because of its file size. The repository instead preserves analytical code, the notebook, models, outputs, the dashboard, and documentation.
7. **No official government endorsement.** CivicPulse AI is an independent academic/portfolio project. It does not represent or claim endorsement by NYC government, NYC 311, AICTE, IBM, BharatCares, or any other government or institutional authority unless explicitly stated otherwise.

---

## Responsible Use

CivicPulse AI should be used as an **analytical exploration and decision-support prototype**, not as a replacement for official government information, emergency services, infrastructure inspections, or public-sector decision-making. The project demonstrates how historical citizen service-request data can be transformed into structured analytical signals.

---

## Project Outcomes

CivicPulse AI brings together several analytical disciplines in a single system:

```text
Data Analytics + Natural Language Processing + Machine Learning
+ Geospatial Intelligence + Time-Series Analysis
+ Interactive Visualization + Explainable Insights
```

The result is an end-to-end urban intelligence workflow rather than an isolated machine-learning model.

---

## Academic / Portfolio Context

**CivicPulse AI** was developed as an academic and portfolio project under:

**AICTE · IBM SkillsBuild Data Analytics with AI Internship 2026 · BharatCares**

The project demonstrates practical application of data analytics, machine learning, NLP, geospatial analysis, data visualization, dashboard engineering, model interpretation, and reproducible analytical workflows.

---

## Future Enhancements

- Live NYC 311 data ingestion
- Automated scheduled data refresh
- Larger historical datasets
- Real-time hotspot monitoring
- Advanced time-series models
- Transformer-based NLP classification
- More granular geographic analysis
- Interactive model explainability
- Automated anomaly detection
- Multi-city urban intelligence
- API-based architecture
- Cloud deployment
- Role-based civic analytics workspaces
- Automated evidence reports

---

## Project Status

**Status: Completed academic/portfolio prototype**

The repository contains a complete analytical notebook, trained model artifacts, generated analytical outputs, an interactive dashboard, a Streamlit reference application, project documentation, and a reproducible analytical workflow.

---

## Author

**Riya Mandal**
CivicPulse AI — Urban Problem Intelligence Engine

GitHub: [https://github.com/imriyamandal/CivicPulse-AI](https://github.com/imriyamandal/CivicPulse-AI)

---

## License

This project is intended for academic, educational and portfolio purposes. See the repository's `LICENSE` file for the applicable license terms.

---

## Acknowledgements

The project builds upon publicly available NYC 311 service-request data and open-source Python/data-science technologies, including Python, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Folium, Streamlit, Jupyter, and Chart.js.

---

<p align="center">

### CivicPulse AI

**Turning citizen reports into structured urban intelligence.**

**Complaints → Patterns → Locations → Signals → Predictions → Insights**

</p>