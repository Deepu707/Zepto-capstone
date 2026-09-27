"""
Module 1 — Data Pipeline
Scrapes books.toscrape.com across 3 categories, cleans the fields,
converts GBP -> INR at the project's fixed baseline rate,
and loads into a normalized SQLite database.

Run:  python scrape.py
Outputs:
  - zepto_books.db          (SQLite database with categories + books tables)
  - books_clean.csv         (optional intermediate CSV for inspection)
"""

import re
import sqlite3
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = urljoin(BASE_URL, "catalogue/category/books_1/index.html")

# Project-defined fixed baseline conversion rate (NOT a live market rate).
GBP_TO_INR = 105.50

# We'll scrape these 3 categories (URL fragments found on the site's sidebar).
# Names here must match the breadcrumb text exactly, else the script will
# fall back to whatever name the breadcrumb shows.
TARGET_CATEGORIES = ["Travel", "Mystery", "Historical Fiction"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CapstoneScraper/1.0; "
        "+https://example.com/bot)"
    )
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a parsed BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_category_url(category_name: str) -> str:
    """Locate the catalogue URL for a category by name from the sidebar."""
    soup = get_soup(BASE_URL)
    sidebar = soup.find("div", class_="side_categories")
    for a in sidebar.find_all("a"):
        name = a.get_text(strip=True)
        if name.lower() == category_name.lower():
            return urljoin(BASE_URL, a["href"])
    raise ValueError(f"Category not found on sidebar: {category_name}")


def parse_book_card(card, category_name: str) -> dict:
    """Extract one book's raw fields from its <article class='product_pod'>."""
    title = card.h3.a["title"].strip()

    price_text = card.find("p", class_="price_color").get_text(strip=True)
    # e.g. "£51.77"

    rating_p = card.find("p", class_="star-rating")
    # class list = ["star-rating", "Three"]
    rating_text = [c for c in rating_p["class"] if c != "star-rating"][0]

    availability_text = card.find(
        "p", class_="instock availability"
    ).get_text(strip=True)

    return {
        "title": title,
        "price_raw": price_text,
        "star_rating_text": rating_text,
        "availability_raw": availability_text,
        "category": category_name,
    }


def scrape_category(category_name: str, max_pages: int = 10) -> list[dict]:
    """Scrape every page of one category. max_pages is a safety cap."""
    url = find_category_url(category_name)
    rows = []
    page = 1
    while url and page <= max_pages:
        soup = get_soup(url)
        cards = soup.find_all("article", class_="product_pod")
        for c in cards:
            rows.append(parse_book_card(c, category_name))

        # Next page?
        next_btn = soup.find("li", class_="next")
        if next_btn and next_btn.a:
            url = urljoin(url, next_btn.a["href"])
            page += 1
            time.sleep(0.5)  # polite delay
        else:
            url = None
    return rows


def scrape_all() -> pd.DataFrame:
    """Scrape all target categories and return a raw DataFrame."""
    all_rows = []
    for cat in TARGET_CATEGORIES:
        print(f"  -> scraping category: {cat}")
        all_rows.extend(scrape_category(cat))
    df = pd.DataFrame(all_rows)
    print(f"Scraped {len(df)} raw rows across {df['category'].nunique()} categories.")
    return df


# ----------------------------- Cleaning -----------------------------

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean scraped fields into proper types.
    Parsing failures on numeric columns are median-imputed (a small, robust
    choice that keeps row count and avoids dropping sparse data).
    Parsing failures on categorical/boolean columns drop the row, since
    median-imputing a category or stock flag is meaningless.
    """
    df = df.copy()

    # price_gbp: strip non-numeric chars, coerce to float
    df["price_gbp"] = (
        df["price_raw"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )

    # rating: map text -> int 1..5
    df["rating"] = df["star_rating_text"].map(RATING_MAP).astype("Int64")

    # in_stock: substring "In stock" -> True, else False
    df["in_stock"] = (
        df["availability_raw"].str.contains("In stock", case=False, na=False)
    )

    # --- failure handling ---
    n_bad_price = df["price_gbp"].isna().sum()
    n_bad_rating = df["rating"].isna().sum()
    n_bad_title = df["title"].isna().sum() | (df["title"].str.strip() == "").sum()

    if n_bad_price:
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"    median-imputed {n_bad_price} bad price rows -> {median_price:.2f}")

    if n_bad_rating:
        median_rating = int(df["rating"].dropna().median())
        df["rating"] = df["rating"].fillna(median_rating).astype(int)
        print(f"    median-imputed {n_bad_rating} bad rating rows -> {median_rating}")

    # drop rows with broken categorical / boolean fields
    before = len(df)
    df = df.dropna(subset=["title", "category"])
    df = df[df["title"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"    dropped {dropped} rows with unparseable title/category")

    # rating back to plain int
    df["rating"] = df["rating"].astype(int)

    # INR conversion at the project's fixed baseline rate
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    # keep only the final required columns, in order
    df = df[
        ["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]
    ].reset_index(drop=True)

    return df


# ----------------------------- DB Load -----------------------------

DB_SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def load_to_sqlite(df: pd.DataFrame, db_path: str = "zepto_books.db") -> None:
    """Create the schema and insert cleaned data."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(DB_SCHEMA)

    # categories
    categories = sorted(df["category"].unique().tolist())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?);",
        [(c,) for c in categories],
    )
    cat_id = {c: i + 1 for i, c in enumerate(categories)}  # AUTOINCREMENT order

    # books
    book_rows = [
        (
            r.title,
            float(r.price_gbp),
            float(r.price_inr),
            int(r.rating),
            int(bool(r.in_stock)),
            cat_id[r.category],
        )
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books
           (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?);""",
        book_rows,
    )
    con.commit()

    n_cats = cur.execute("SELECT COUNT(*) FROM categories;").fetchone()[0]
    n_books = cur.execute("SELECT COUNT(*) FROM books;").fetchone()[0]
    con.close()
    print(f"Loaded {n_books} books across {n_cats} categories into {db_path}")


# ----------------------------- Main -----------------------------

def main():
    print("Step 1: scrape")
    raw = scrape_all()
    raw.to_csv("books_raw.csv", index=False)

    print("Step 2: clean + convert")
    clean_df = clean(raw)
    clean_df.to_csv("books_clean.csv", index=False)
    print(f"  clean rows: {len(clean_df)}")
    print(f"  fixed conversion: 1 GBP = {GBP_TO_INR} INR")

    print("Step 3: load to SQLite")
    load_to_sqlite(clean_df, "zepto_books.db")

    print("Done.")


if __name__ == "__main__":
    main()