# Live Customer Churn Risk Assessment Engine
### With Real-Time Recommendation Overrides for E-Commerce

> **Academic Project** | Designed for clean evolution into a production-grade system.

---

## Overview

This project implements a **Live Customer Churn Risk Assessment Engine** that continuously estimates churn probability from real-time behavioural events.  
When the churn probability surpasses a configurable threshold, a **middleware Decision Gate** overrides the standard recommendation engine and injects retention-focused recommendations.

```
Live Behavioural Events
        │
        ▼
┌───────────────────┐
│   Event Stream    │  ← real-time ingestion layer
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing /  │  ← cleaning, feature engineering
│  Feature Eng.     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Churn Predictor  │  ← ML model inference
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Decision Gate    │  ← threshold-based override trigger
└────────┬──────────┘
         │ High Risk             │ Low Risk
         ▼                       ▼
┌────────────────┐     ┌──────────────────────┐
│ Override Engine│     │ Normal Recommender   │
│ (Retention)    │     │                      │
└────────────────┘     └──────────────────────┘
```

---

## Project Structure

```
churn-engine/
├── data/                   # Raw and processed datasets (not committed)
├── models/                 # Serialised ML artefacts
├── logs/                   # Runtime log files
├── configs/
│   ├── config.yaml         # Main configuration
│   └── logging.yaml        # Logging configuration
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning pipeline
│   ├── feature_engineering.py
│   ├── train_model.py      # Model training workflow
│   ├── predictor.py        # Churn score inference
│   ├── event_stream.py     # Live event ingestion
│   ├── decision_gate.py    # Threshold decision logic
│   ├── recommender.py      # Standard recommendation engine
│   ├── override_engine.py  # Retention recommendation override
│   ├── evaluation.py       # Model evaluation utilities
│   └── utils.py            # Shared utility functions
├── dashboard/
│   ├── pages/              # Streamlit multi-page modules
│   └── assets/             # Static assets (images, CSS)
├── tests/                  # pytest test suite
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Tech Stack

| Category        | Library / Tool                  |
|-----------------|---------------------------------|
| Language        | Python 3.10+                    |
| Data            | pandas, numpy, scipy            |
| ML              | scikit-learn, xgboost, joblib   |
| Visualisation   | matplotlib, seaborn, plotly     |
| Dashboard       | Streamlit                       |
| Config          | PyYAML, python-dotenv, pydantic |
| Logging         | logging, colorlog               |
| Testing         | pytest, pytest-cov              |

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd churn-engine
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Directory Scaffolding

The following directories should exist (created automatically by the app):

```
data/   models/   logs/   configs/
```

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

To connect Live Storefront Activity Feed to the deployed backend, set the
Streamlit secret `API_BASE_URL` to the deployed HTTPS backend URL. In
production, the dashboard requires this secret and does not fall back to
localhost. For local development, omit the secret to use the default
`http://127.0.0.1:8000` backend.

---

## Phase Roadmap

| Phase | Description                                          | Status        |
|-------|------------------------------------------------------|---------------|
| **1** | Project foundation, architecture, placeholder modules | ✅ Complete   |
| **2** | Data ingestion, preprocessing, feature engineering   | 🔲 Planned    |
| **3** | Model training — XGBoost churn classifier            | 🔲 Planned    |
| **4** | Live event stream integration & real-time inference  | 🔲 Planned    |
| **5** | Decision Gate + Override Engine                      | 🔲 Planned    |
| **6** | Recommendation Engine (normal + retention)           | 🔲 Planned    |
| **7** | Evaluation dashboard & reporting                     | 🔲 Planned    |

---

## Configuration

Edit `configs/config.yaml` to adjust:

- **churn threshold** — probability above which override triggers
- **model paths** — where serialised models are stored
- **data paths** — locations of raw and processed data
- **logging level** — DEBUG / INFO / WARNING

### Production checkout storage (Render)

Checkout finalizations and redeemed product offers are stored transactionally
in SQLite. Configure a writable absolute path with an existing parent directory:

```text
CHURN_ENVIRONMENT=production
CHURN_LOCAL_DEMO_MODE=false
CORS_ORIGINS=https://<published-storefront-origin>
CHECKOUT_LEDGER_PATH=/tmp/checkout_ledger.sqlite3
```

The API refuses production startup if the demo flag, public CORS origins, or
ledger path is missing or invalid. A path under `/tmp` works on a Render free
web service, but its contents are ephemeral: checkout finalization and offer
redemption records can be lost when the service restarts or is redeployed.
After such a loss, idempotency records and one-time offer redemption history
are not recoverable. Use a Render persistent disk or shared transactional
database when durable records are required. SQLite supports transactional
retries and concurrent requests for a single backend instance; do not scale
multiple instances or workers against one SQLite ledger.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Licence

MIT — for academic and research use.
