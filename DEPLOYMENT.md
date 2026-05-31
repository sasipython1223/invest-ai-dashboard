# Deployment Guide — Streamlit Community Cloud

This guide explains how to deploy **invest-ai-dashboard** as a live Streamlit web app so you can access it from a mobile browser.

> **Guardrail:** This dashboard is an educational prototype only. Not financial advice. No auto-trading, broker API, or Tiger API execution. All trades must be manually reviewed and executed by the user.

---

## Main file path

```
app/dashboard.py
```

---

## Deploy to Streamlit Community Cloud

### Prerequisites

- A GitHub account with access to the repository.
- A [Streamlit Community Cloud](https://streamlit.io/cloud) account (free tier is sufficient).

### Steps

1. **Push latest code to `main`.**

   ```bash
   git push origin main
   ```

2. **Go to Streamlit Cloud.**

   Open [https://share.streamlit.io](https://share.streamlit.io) in your browser.

3. **Create a new app.**

   Click **New app**.

4. **Select the GitHub repository.**

   ```
   sasipython1223/invest-ai-dashboard
   ```

5. **Set the branch.**

   ```
   main
   ```

6. **Set the main file path.**

   ```
   app/dashboard.py
   ```

7. **Deploy.**

   Click **Deploy**. Streamlit will install dependencies from `requirements.txt` and start the app.

8. **Open the generated URL on your mobile browser.**

   After deployment succeeds, Streamlit shows a public URL such as:
   ```
   https://<your-app-name>.streamlit.app
   ```
   Open that URL on your phone.

---

## Secrets and API keys

### How secrets work on Streamlit Cloud

Streamlit Cloud provides a **Secrets** panel (Settings → Secrets) where you can paste key-value pairs. These are injected as environment variables at runtime and are **never stored in the repository**.

### Setting secrets on Streamlit Cloud

In the Streamlit Cloud dashboard, go to **App → Settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your-actual-gemini-api-key-here"
```

> **Do not commit real API keys to the repository.** Use `.streamlit/secrets.example.toml` as a reference only.

### Optional keys

| Variable | Purpose | Required |
|---|---|---|
| `GEMINI_API_KEY` | Enables Gemini AI review | Optional |
| `OPENAI_API_KEY` | Reserved for future OpenAI integration | Optional |

If a key is absent, the corresponding AI reviewer shows a disabled / placeholder message in the UI. The rest of the dashboard continues to work.

### Local development

Copy `.env.example` to `.env` and fill in any keys you want to use locally:

```bash
cp .env.example .env
# Edit .env and add your keys
```

See `.streamlit/secrets.example.toml` for the Streamlit-format equivalent.

---

## Mobile use notes

- **Best viewed in mobile browser** — the app uses a wide layout and renders charts at full width.
- **Use landscape mode** for wide tables and charts when screen space is limited.
- **No trades are executed by this app** — it is a decision-support tool only.
- **Always verify prices in your broker app** before placing any manual order.
- The app requires an active internet connection to fetch live price data.

---

## Smoke-test checklist

After deployment, verify the following manually:

- [ ] App loads without import errors.
- [ ] Overview tab loads.
- [ ] Portfolio tab loads.
- [ ] Portfolio Outcome Simulator loads.
- [ ] Ticker Review loads.
- [ ] Gemini review gracefully shows disabled / placeholder / failure message if key is absent.
- [ ] No secrets are printed in the UI or logs.
- [ ] Mobile browser can open the app URL.

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Add keys if needed
streamlit run app/dashboard.py
```

---

## Running tests

```bash
pytest -q
```
