# Pre-Delinquency Intervention Engine 
hi
An end-to-end, ML-powered credit risk intelligence platform for **early delinquency detection**, **customer-level intervention**, and **portfolio monitoring**.

This repository contains:

- **Backend**: Python Azure Functions + Azure Cosmos DB
- **Frontend**: React + TypeScript dashboard (`risk-guard-dashboard/`)
- **Modeling**: Sequence-based risk inference (LSTM) + multi-layer risk enrichment
- **Realtime updates**: Server-Sent Events (SSE) bridge for live dashboard refresh

---

## Key Capabilities

- Customer-level risk scoring (PD, TTC, LSI, ECL)
- Portfolio-level aggregations and IFRS9-style stage summaries
- On-demand re-scoring (`/api/score/{customerId}`)
- Scenario stress analysis per customer
- Model diagnostics and metrics endpoint
- Live UI synchronization via SSE (`/events`)

---

## Architecture Overview

```text
Frontend (React + Vite)
      |
      |  /api/* (proxied)
      v
Azure Functions (Python, port 7071)
  - risk_scores
  - risk_scores_by_id
  - customers
  - portfolio_summary
  - stress_test
  - rescore
  - model_metrics
      |
      v
Azure Cosmos DB (Emulator/local or cloud)

SSE Server (FastAPI, port 7072)
  - polls Cosmos changes
  - pushes /events to frontend
```

---

## Repository Structure

```text
hoh/
├─ customers/                  # GET /api/customers
├─ risk_scores/               # GET /api/risk-scores
├─ risk_scores_by_id/         # GET /api/risk-scores/{customerId}
├─ portfolio_summary/         # GET /api/portfolio/summary
├─ stress_test/               # GET /api/risk-scores/{customerId}/stress
├─ rescore/                   # POST /api/score/{customerId}
├─ model_metrics/             # GET /api/model/metrics
├─ _shared/                   # shared backend utils/loaders
├─ models/                    # model artifacts
├─ scripts/                   # training and utility scripts
├─ sse_server.py              # FastAPI SSE bridge
├─ host.json
├─ local.settings.json
├─ requirements.txt
└─ risk-guard-dashboard/      # React frontend
```

---

## Tech Stack

### Backend
- Python 3.x
- Azure Functions (HTTP triggers)
- Azure Cosmos DB SDK
- FastAPI + Uvicorn (SSE server)

### ML/Data
- PyTorch (LSTM)
- scikit-learn, XGBoost, LightGBM, CatBoost
- pandas, numpy, scipy

### Frontend
- React 18 + TypeScript
- Vite
- TanStack Query
- Recharts
- Tailwind CSS + shadcn/ui

---

## Local Setup

### 1) Prerequisites

- Python (recommended 3.10+)
- Node.js 18+
- Azure Functions Core Tools
- Azure Cosmos DB Emulator (if running locally)

### 2) Install backend dependencies

```bash
cd hoh
pip install -r requirements.txt
```

### 3) Install frontend dependencies

```bash
cd risk-guard-dashboard
npm install
```

---

## Run the Project (3 terminals)

### Terminal A — Azure Functions API

```bash
cd hoh
func host start
```

- Base URL: `http://localhost:7071`

### Terminal B — SSE Server

```bash
cd hoh
uvicorn sse_server:app --host 0.0.0.0 --port 7072
```

- Health: `http://localhost:7072/health`
- Stream: `http://localhost:7072/events`

### Terminal C — Frontend

```bash
cd hoh/risk-guard-dashboard
npm run dev
```

- App URL: `http://localhost:8080`

---

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/risk-scores` | Paginated/filterable risk score list |
| GET | `/api/risk-scores/{customerId}` | Single customer risk profile |
| GET | `/api/customers` | Customer profile list/details |
| GET | `/api/portfolio/summary` | Portfolio KPIs + stage/tier summaries |
| GET | `/api/risk-scores/{customerId}/stress` | Customer stress-test results |
| GET | `/api/model/metrics` | Model diagnostics/metrics |
| POST | `/api/score/{customerId}` | Trigger re-inference and upsert |

---

## ML / Risk Pipeline

Typical scoring flow:

1. Load customer sequence + profile context
2. Run model inference (LSTM or fallback scorer)
3. Apply risk enrichment and macro adjustments
4. Compute derived metrics (PD variants, LSI, ECL, etc.)
5. Assign risk tier + stage
6. Persist to Cosmos and broadcast update (SSE)

---

## Frontend Modules

- **Dashboard**: KPI cards, risk distributions, top-risk customers
- **Watch List**: triage table, filters, customer deep-links
- **Customer Detail**: rich single-customer risk dossier
- **Portfolio**: scenario and distribution analytics
- **Stress Test**: what-if simulation views
- **Model Observatory**: model metrics and feature-level insights

---

## Testing & Quality

### Frontend

```bash
cd risk-guard-dashboard
npm run test
npm run lint
npm run build
```

### Backend

- Validate endpoints locally via Functions host
- Verify Cosmos connectivity and seeded data
- Confirm `/health` and `/events` stream from SSE server

---

## Configuration & Security Notes

- `local.settings.json` is for local development only.
- Do **not** commit real secrets/keys in production workflows.
- Use environment variables or secret managers (Key Vault) for sensitive values.
- Keep Cosmos keys, API secrets, and external service credentials rotated.

---

## Troubleshooting

### Re-score fails with "No sequence data found"
- Ensure sequence data exists in source file/container for that customer.
- If fallback logic is enabled, verify `rescore` function is up-to-date.

### Frontend not updating live
- Ensure `sse_server.py` is running on `7072`.
- Ensure Vite proxy forwards `/events` correctly.

### API calls failing from frontend
- Ensure Azure Functions host is running on `7071`.
- Confirm `host.json` route prefix is `api`.

---

## Roadmap

- Add auth + role-based access control
- Add CI/CD and environment promotion
- Expand model observability (drift dashboards, alerting)
- Harden lint/type rules and UI accessibility baseline

---

## License

This project is maintained by the repository owner. Add an explicit OSS/commercial license as needed.
