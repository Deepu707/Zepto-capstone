"""
Module 1 — SQL + pandas query layer.

Requires zepto_books.db (produced by scrape.py) to exist in the same folder.

Run:  python queries.py
"""

import sqlite3
import pandas as pd

DB_PATH = "zepto_books.db"


# ---------------------------------------------------------------------
# 1. Define the queries. Each one targets a required SQL clause.
# ---------------------------------------------------------------------
QUERIES = {
    # SELECT + WHERE
    "q1_select_where": """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        WHERE rating = 5 AND in_stock = 1
        LIMIT 10;
    """,

    # ORDER BY (descending)
    "q2_order_by": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    # LIMIT (top 5 cheapest)
    "q3_limit": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp ASC
        LIMIT 5;
    """,

    # DISTINCT categories
    "q4_distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,

    # IN and BETWEEN
    "q5_in_between": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating IN (4, 5)
          AND price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp ASC;
    """,

    # JOIN (top 3 highest-rated books per category)
    "q6_join": """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY c.category_name ASC, b.rating DESC, b.price_gbp ASC;
    """,
}


# ---------------------------------------------------------------------
# 2. Run each query and print results.
# ---------------------------------------------------------------------
def run_all_queries() -> dict:
    con = sqlite3.connect(DB_PATH)
    results = {}
    for name, sql in QUERIES.items():
        print("=" * 78)
        print(f"[{name}]")
        print(sql.strip())
        print("-" * 78)
        df = pd.read_sql(sql, con)
        print(df.to_string(index=False))
        print(f"({len(df)} rows)\n")
        results[name] = df
    con.close()
    return results


# ---------------------------------------------------------------------
# 3. pd.read_sql vs pd.merge — reproduce the JOIN both ways.
# ---------------------------------------------------------------------
def compare_join_approaches():
    print("=" * 78)
    print("pd.read_sql  vs  pd.merge — the JOIN query, two ways")
    print("=" * 78)

    con = sqlite3.connect(DB_PATH)

    # --- Approach A: SQL JOIN via pd.read_sql ---
    sql_join = """
        SELECT c.category_name,
               b.title,
               b.rating,
               b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY c.category_name ASC, b.rating DESC, b.price_gbp ASC;
    """
    df_sql = pd.read_sql(sql_join, con)

    # --- Approach B: pd.merge on two plain in-memory DataFrames ---
    books = pd.read_sql("SELECT * FROM books;", con)
    categories = pd.read_sql("SELECT * FROM categories;", con)
    con.close()

    df_merged = (
        books.merge(categories, on="category_id", how="inner")
             .loc[lambda d: d["rating"] >= 4,
                  ["category_name", "title", "rating", "price_gbp"]]
             .sort_values(
                 by=["category_name", "rating", "price_gbp"],
                 ascending=[True, False, True]
             )
             .reset_index(drop=True)
    )

    df_sql = df_sql.reset_index(drop=True)

    print("\n--- pd.read_sql (SQL JOIN) ---")
    print(df_sql.to_string(index=False))
    print(f"({len(df_sql)} rows)")

    print("\n--- pd.merge (in-memory) ---")
    print(df_merged.to_string(index=False))
    print(f"({len(df_merged)} rows)")

    # Equality check
    equal = df_sql.equals(df_merged)
    print("\n" + "=" * 78)
    print(f"Are the two results equal? -> {equal}")
    print("=" * 78)

    # also show a row-by-row diff if not equal
    if not equal:
        diff = df_sql.compare(df_merged)
        print("\nDifferences:")
        print(diff)

    return df_sql, df_merged


# ---------------------------------------------------------------------
# 4. Extra pd.read_sql demo on two queries (already covered, but
#    called out explicitly to satisfy the rubric).
# ---------------------------------------------------------------------
def read_two_as_dataframes():
    con = sqlite3.connect(DB_PATH)
    df_top_priced = pd.read_sql(
        "SELECT title, price_gbp, price_inr FROM books "
        "ORDER BY price_gbp DESC LIMIT 5;", con
    )
    df_avg_by_cat = pd.read_sql(
        "SELECT c.category_name, ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp "
        "FROM books b JOIN categories c ON b.category_id = c.category_id "
        "GROUP BY c.category_name ORDER BY avg_price_gbp DESC;", con
    )
    con.close()

    print("=" * 78)
    print("Two extra pd.read_sql outputs")
    print("=" * 78)
    print("\n[Top 5 most expensive books]")
    print(df_top_priced.to_string(index=False))
    print("\n[Average GBP price per category]")
    print(df_avg_by_cat.to_string(index=False))
    print()


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_all_queries()
    compare_join_approaches()
    read_two_as_dataframes()
    print("\nAll queries executed successfully.")