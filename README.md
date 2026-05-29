# invest-ai-dashboard

Personal AI-assisted investment decision dashboard for long-term and tactical opportunities.

## MVP status
This repository contains the initial MVP skeleton.

## Core principle
**Rules decide. AI explains. User executes manually.**

## Guardrails
- No auto-trading
- No broker order placement API integration
- No AI-generated raw buy/sell signals
- Deterministic, rule-based, testable signals only
- User executes trades manually

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
streamlit run app/dashboard.py
```

## Data files
- `data/watchlist.csv`
- `data/portfolio.csv`
- `data/trade_journal.csv`

## Disclaimer
This project is for educational decision-support prototyping only. It is **not financial advice**. Any trade must be manually reviewed and executed by the user.
