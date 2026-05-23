import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
SEED_PATH = BASE_DIR / "dev-seeds.json"
OUT_DIR = BASE_DIR / "dwh_mock"

RANDOM_SEED = 42
TOTAL_INTERACTIONS = 5000
SALES_RATIO = 0.6
CART_RATIO = 0.2
REVIEW_RATIO = 0.2

TRAIN_MONTH = 10
VALIDATION_MONTH = 11
TEST_MONTH = 12

MONTH_WEIGHTS = [0.7, 0.15, 0.15]
RATING_WEIGHTS = [0.05, 0.05, 0.2, 0.35, 0.35]


def _ascii_text(value: str, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    # Keep Unicode (Vietnamese accents) and only normalize whitespace.
    cleaned = unicodedata.normalize("NFC", str(value))
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _random_datetime(year: int, month: int) -> datetime:
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(year, month, day, hour, minute, second)


def _choose_month() -> int:
    return random.choices(
        [TRAIN_MONTH, VALIDATION_MONTH, TEST_MONTH],
        weights=MONTH_WEIGHTS,
        k=1,
    )[0]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    random.seed(RANDOM_SEED)

    with SEED_PATH.open("r", encoding="utf-8") as f:
        seeds = json.load(f)

    buyers = seeds.get("buyers", [])
    sellers = seeds.get("sellers", [])
    products = seeds.get("products", [])

    buyer_ids = {b["username"]: idx + 1 for idx, b in enumerate(buyers)}
    seller_ids = {s["username"]: idx + 1 for idx, s in enumerate(sellers)}

    dim_users = []
    for buyer in buyers:
        dim_users.append(
            {
                "user_id": buyer_ids[buyer["username"]],
                "username": _ascii_text(buyer["username"], "user"),
                "full_name": _ascii_text(buyer.get("full_name", ""), "Unknown"),
                "email": _ascii_text(buyer.get("email", ""), "unknown@example.com"),
                "role": "buyer",
                "is_active": bool(buyer.get("is_active", True)),
            }
        )

    books = []
    for idx, product in enumerate(products, start=1):
        books.append(
            {
                "book_id": idx,
                "title": _ascii_text(product.get("title", ""), "Untitled"),
                "author": _ascii_text(product.get("author", ""), "Unknown"),
                "category_name": _ascii_text(product.get("category_name", ""), "Unknown"),
                "price": float(product.get("price", 0.0)),
                "purchase_count": 0,
                "rating_avg": 0.0,
                "seller_username": _ascii_text(product.get("seller_username", ""), "unknown"),
            }
        )

    book_id_by_title = {product.get("title"): idx + 1 for idx, product in enumerate(products)}

    year = datetime.now().year - 1  # BUG FIX: dùng năm trước để tránh months_ago âm

    sales_count = int(TOTAL_INTERACTIONS * SALES_RATIO)
    cart_count = int(TOTAL_INTERACTIONS * CART_RATIO)
    review_count = TOTAL_INTERACTIONS - sales_count - cart_count

    fact_sales = []
    sales_pairs = []
    for i in range(1, sales_count + 1):
        buyer = random.choice(buyers)
        product = random.choice(products)
        book_id = book_id_by_title.get(product.get("title"))
        if not book_id:
            continue

        buyer_id = buyer_ids[buyer["username"]]
        seller_id = seller_ids.get(product.get("seller_username"), 1)

        order_date = _random_datetime(year, _choose_month())
        quantity = random.randint(1, 3)
        unit_price = float(product.get("price", 0.0))
        line_total = round(unit_price * quantity, 2)

        fact_sales.append(
            {
                "order_item_id": i,
                "order_id": i,
                "buyer_id": buyer_id,
                "seller_id": seller_id,
                "book_id": book_id,
                "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "line_total": line_total,
                "order_overall_status": "completed",
                "item_status": "completed",
                "payment_method": random.choice(["vnpay", "cod", "momo"]),
                "payment_status": "paid",
            }
        )
        sales_pairs.append((buyer_id, book_id))
        books[book_id - 1]["purchase_count"] += 1

    fact_cart = []
    for i in range(1, cart_count + 1):
        buyer = random.choice(buyers)
        product = random.choice(products)
        book_id = book_id_by_title.get(product.get("title"))
        if not book_id:
            continue

        buyer_id = buyer_ids[buyer["username"]]
        added_at = _random_datetime(year, _choose_month())
        quantity = random.randint(1, 2)

        fact_cart.append(
            {
                "cart_id": i,
                "buyer_id": buyer_id,
                "book_id": book_id,
                "quantity": quantity,
                "added_at": added_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    fact_reviews = []
    review_counts = defaultdict(int)
    review_scores_sum = defaultdict(float)

    for i in range(1, review_count + 1):
        if sales_pairs and random.random() < 0.8:
            buyer_id, book_id = random.choice(sales_pairs)
        else:
            buyer = random.choice(buyers)
            product = random.choice(products)
            book_id = book_id_by_title.get(product.get("title"))
            if not book_id:
                continue
            buyer_id = buyer_ids[buyer["username"]]

        score = random.choices([1, 2, 3, 4, 5], weights=RATING_WEIGHTS, k=1)[0]
        snapshot_date = _random_datetime(year, _choose_month())

        review_counts[book_id] += 1
        review_scores_sum[book_id] += score

        fact_reviews.append(
            {
                "book_id": book_id,
                "buyer_id": buyer_id,
                "score": float(score),
                "total_reviews_at_snapshot": review_counts[book_id],
                "snapshot_date": snapshot_date.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    for book_id, count in review_counts.items():
        avg = review_scores_sum[book_id] / max(count, 1)
        books[book_id - 1]["rating_avg"] = round(avg, 2)

    dim_date_rows = {}
    for row in fact_sales:
        dt = datetime.strptime(row["order_date"], "%Y-%m-%d %H:%M:%S").date()
        dim_date_rows[dt] = dt
    for row in fact_cart:
        dt = datetime.strptime(row["added_at"], "%Y-%m-%d %H:%M:%S").date()
        dim_date_rows[dt] = dt
    for row in fact_reviews:
        dt = datetime.strptime(row["snapshot_date"], "%Y-%m-%d %H:%M:%S").date()
        dim_date_rows[dt] = dt

    dim_date = []
    for dt in sorted(dim_date_rows.values()):
        dim_date.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "day_of_week": dt.strftime("%A"),
                "quarter": (dt.month - 1) // 3 + 1,
                "is_weekend": dt.weekday() >= 5,
            }
        )

    _write_csv(
        OUT_DIR / "dim_users.csv",
        ["user_id", "username", "full_name", "email", "role", "is_active"],
        dim_users,
    )
    _write_csv(
        OUT_DIR / "dim_books.csv",
        [
            "author",
            "book_id",
            "category_name",
            "price",
            "purchase_count",
            "rating_avg",
            "seller_username",
            "title",
        ],
        books,
    )
    _write_csv(
        OUT_DIR / "fact_sales.csv",
        [
            "order_item_id",
            "order_id",
            "buyer_id",
            "seller_id",
            "book_id",
            "order_date",
            "quantity",
            "unit_price",
            "line_total",
            "order_overall_status",
            "item_status",
            "payment_method",
            "payment_status",
        ],
        fact_sales,
    )
    _write_csv(
        OUT_DIR / "fact_cart.csv",
        ["cart_id", "buyer_id", "book_id", "quantity", "added_at"],
        fact_cart,
    )
    _write_csv(
        OUT_DIR / "fact_reviews.csv",
        ["book_id", "buyer_id", "score", "total_reviews_at_snapshot", "snapshot_date"],
        fact_reviews,
    )
    _write_csv(
        OUT_DIR / "dim_date.csv",
        ["date", "year", "month", "day", "day_of_week", "quarter", "is_weekend"],
        dim_date,
    )

    print("Wrote mock DWH data to", OUT_DIR)
    print("fact_sales:", len(fact_sales))
    print("fact_cart:", len(fact_cart))
    print("fact_reviews:", len(fact_reviews))


if __name__ == "__main__":
    main()
