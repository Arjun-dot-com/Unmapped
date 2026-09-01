from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "platform" / "backend" / "data"
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "flights.sqlite3"
tasks = {}

def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS flights(id TEXT PRIMARY KEY, name TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, root TEXT)")
    c.commit()
    return c
