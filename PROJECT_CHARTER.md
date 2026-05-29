# PROJECT_CHARTER

## Project
invest-ai-dashboard

## Purpose
Build a personal AI-assisted investment decision dashboard for long-term growth and selective short-term tactical opportunities.

## Core Principle
Rules decide. AI explains. User executes manually.

## Guardrails
- No auto-trading.
- No broker order placement API integration.
- No AI-generated raw buy/sell signals.
- Signals must be deterministic, rule-based, and testable.
- AI can only explain/challenge rule outputs.
- All execution remains manual.
- Recommendations must include signal reason, risk status, and human checklist.
- Secrets must be provided through environment variables.
