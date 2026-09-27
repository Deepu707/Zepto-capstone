# Module 1 — Data Pipeline

A raw-to-relational ETL pipeline that scrapes live book data from
[books.toscrape.com](https://books.toscrape.com/), cleans the fields into
proper types, converts GBP prices to INR using the project's fixed baseline
rate, loads everything into a normalized SQLite database, and demonstrates
querying it both via SQL and via pandas.

---

## Deliverables in this folder

| File | Purpose |
|---|---|
| `scrape.py` | Scraper + cleaner + SQLite loader (end-to-end pipeline) |
| `queries.py` | 6 SQL queries + `pd.read_sql` vs `pd.merge` comparison |
| `zepto_books.db` | SQLite database with `categories` + `books` tables (69 rows) |
| `books_raw.csv` | Raw scraped rows (pre-clean), for inspection |
| `books_clean.csv` | Cleaned rows (post-clean, pre-DB), for inspection |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt