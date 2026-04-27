# Greek DAM Price Forecasting

End-to-end MLOps project for forecasting Greek day-ahead electricity market
(DAM) prices using ENTSO-E data, deployed on Azure Machine Learning.

## Project structure

- `src/` — production code (data loaders, features, training, inference)
- `notebooks/` — exploratory analysis (not deployed)
- `data/raw/` — raw data (gitignored; mock data is generated on the fly)
- `requirements.in` — direct dependencies
- `requirements.txt` — locked dependencies (auto-generated)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pip-tools
pip-sync requirements.txt
```

Create a `.env` file in the project root:

```
ENTSOE_API_TOKEN=your_token_here
```