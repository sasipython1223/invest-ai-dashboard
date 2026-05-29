import os
from pathlib import Path

from dotenv import load_dotenv


def load_config() -> dict[str, str | None]:
    load_dotenv()
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    return {
        "data_dir": str(data_dir),
        "watchlist_path": str(data_dir / "watchlist.csv"),
        "portfolio_path": str(data_dir / "portfolio.csv"),
        "trade_journal_path": str(data_dir / "trade_journal.csv"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    }
