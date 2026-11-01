from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

#python text_processor.py `
  #--business-path "HW3_data/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json" `
  #--review-path "HW3_data/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json" `
  #--outdir HW3_data

CUTOFF_DATE = date(2021, 1, 1)

BUSINESS_OUTPUT_FIELDS = [
    "business_id",
    "name",
    "categories",
    "city",
    "stars",
    "review_count",
]

REVIEW_OUTPUT_FIELDS = [
    "review_id",
    "user_id",
    "business_id",
    "date",
    "stars",
    "text",
]


def read_jsonl(path: Path, counters: Dict[str, int]) -> Iterable[Dict[str, Any]]:
    """
    Read a JSON Lines file one record at a time.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot open file: {path}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            counters["total_records"] += 1

            if not raw_line.strip():
                counters["empty_lines"] += 1
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                counters["malformed_json"] += 1
                continue

            if not isinstance(record, dict):
                counters["non_object_records"] += 1
                continue

            yield record


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_iso_date(value: Any) -> Optional[date]:
    """
    Accept Yelp timestamps such as '2018-07-07 22:09:11'
    and normalize them to a Python date object.
    Also accepts plain ISO dates such as '2018-07-07'.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    value = value.strip()

    try:
        # Yelp raw review data: YYYY-MM-DD HH:MM:SS
        return datetime.fromisoformat(value).date()
    except ValueError:
        try:
            # Plain calendar date: YYYY-MM-DD
            return date.fromisoformat(value)
        except ValueError:
            return None


def parse_rating(value: Any) -> Optional[float]:
    """
    Parse a valid star rating.
    Yelp ratings are expected to be in [1.0, 5.0].
    """
    if isinstance(value, bool):
        return None

    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None

    if not 1.0 <= rating <= 5.0:
        return None

    return rating


def parse_categories(value: Any) -> list[str]:
    """
    Convert a comma-separated category string to stripped category names.
    Null categories are treated as an empty list.
    """
    if value is None:
        return []

    if not isinstance(value, str):
        return []

    return [
        category.strip()
        for category in value.split(",")
        if category.strip()
    ]


def validate_business(
        record: Dict[str, Any],
        counters: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    required_fields = [
        "business_id",
        "name",
        "city",
        "stars",
        "review_count",
    ]

    for field in required_fields:
        if field not in record or record[field] is None:
            counters[f"missing_business_{field}"] += 1
            return None

    if not is_non_empty_string(record["business_id"]):
        counters["invalid_business_id"] += 1
        return None

    if not is_non_empty_string(record["name"]):
        counters["invalid_business_name"] += 1
        return None

    if not is_non_empty_string(record["city"]):
        counters["invalid_business_city"] += 1
        return None

    if record["city"] != "Philadelphia":
        counters["business_not_philadelphia"] += 1
        return None

    categories = parse_categories(record.get("categories"))

    if "Restaurants" not in categories and "Food" not in categories:
        counters["business_missing_required_category"] += 1
        return None

    stars = parse_rating(record["stars"])
    if stars is None:
        counters["invalid_business_stars"] += 1
        return None

    review_count_value = record["review_count"]

    if isinstance(review_count_value, bool):
        counters["invalid_business_review_count"] += 1
        return None

    if isinstance(review_count_value, int):
        review_count = review_count_value
    elif (
            isinstance(review_count_value, float)
            and review_count_value.is_integer()
    ):
        review_count = int(review_count_value)
    else:
        counters["invalid_business_review_count"] += 1
        return None

    if review_count < 0:
        counters["invalid_business_review_count"] += 1
        return None

    return {
        "business_id": record["business_id"],
        "name": record["name"],
        "categories": record.get("categories"),
        "city": record["city"],
        "stars": stars,
        "review_count": review_count,
    }


def validate_review(
        record: Dict[str, Any],
        retained_business_ids: set[str],
        counters: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    required_fields = [
        "review_id",
        "user_id",
        "business_id",
        "date",
        "stars",
        "text",
    ]

    for field in required_fields:
        if field not in record or record[field] is None:
            counters[f"missing_review_{field}"] += 1
            return None

    if not is_non_empty_string(record["review_id"]):
        counters["invalid_review_id"] += 1
        return None

    if not is_non_empty_string(record["user_id"]):
        counters["invalid_review_user_id"] += 1
        return None

    if not is_non_empty_string(record["business_id"]):
        counters["invalid_review_business_id"] += 1
        return None

    if record["business_id"] not in retained_business_ids:
        counters["review_business_not_retained"] += 1
        return None

    parsed_date = parse_iso_date(record["date"])
    if parsed_date is None:
        counters["invalid_review_date"] += 1
        return None

    stars = parse_rating(record["stars"])
    if stars is None:
        counters["invalid_review_stars"] += 1
        return None

    if not isinstance(record["text"], str):
        counters["invalid_review_text"] += 1
        return None

    return {
        "review_id": record["review_id"].strip(),
        "user_id": record["user_id"].strip(),
        "business_id": record["business_id"].strip(),
        "date": parsed_date,
        "stars": stars,
        "text": record["text"],
    }


def write_jsonl(
        path: Path,
        records: Iterable[Dict[str, Any]],
        fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            output_record = {field: record[field] for field in fields}
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def write_reviews_csv(
        path: Path,
        records: Iterable[Dict[str, Any]],
) -> None:
    fields = [
        "review_id",
        "user_id",
        "business_id",
        "date",
        "stars",
        "text",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for record in records:
            row = dict(record)
            row["date"] = record["date"].isoformat()
            writer.writerow(row)


def prepare_businesses(
        business_path: Path,
        output_path: Path,
        counters: Dict[str, int],
) -> set[str]:
    """
    Filter businesses and write businesses.jsonl.
    """
    businesses_by_id: Dict[str, Dict[str, Any]] = {}

    for record in read_jsonl(business_path, counters):
        business = validate_business(record, counters)

        if business is None:
            continue

        business_id = business["business_id"]

        if business_id in businesses_by_id:
            counters["duplicate_id"] += 1
            continue

        businesses_by_id[business_id] = business

    sorted_businesses = [
        businesses_by_id[business_id]
        for business_id in sorted(businesses_by_id)
    ]

    write_jsonl(
        output_path,
        sorted_businesses,
        BUSINESS_OUTPUT_FIELDS,
    )

    counters["retained"] = len(sorted_businesses)
    return set(businesses_by_id)


def prepare_reviews(
        review_path: Path,
        retained_business_ids: set[str],
        train_output_path: Path,
        dev_output_path: Path,
        qrels_output_path: Path,
        dev_users_output_path: Path,
        review_counters: Dict[str, int],
        split_counters: Dict[str, int],
) -> None:
    """
    Filter, deduplicate, sort, split, and write all review outputs.
    """
    retained_reviews: list[Dict[str, Any]] = []

    for record in read_jsonl(review_path, review_counters):
        review = validate_review(
            record,
            retained_business_ids,
            review_counters,
        )

        if review is not None:
            retained_reviews.append(review)

    # Sorting happens before duplicate removal, as required.
    retained_reviews.sort(
        key=lambda record: (
            record["date"],
            record["review_id"],
        )
    )

    unique_reviews: list[Dict[str, Any]] = []
    seen_review_ids: set[str] = set()

    for review in retained_reviews:
        review_id = review["review_id"]

        if review_id in seen_review_ids:
            review_counters["duplicate_id"] += 1
            continue

        seen_review_ids.add(review_id)
        unique_reviews.append(review)

    train_reviews = [
        review
        for review in unique_reviews
        if review["date"] < CUTOFF_DATE
    ]

    dev_reviews = [
        review
        for review in unique_reviews
        if review["date"] >= CUTOFF_DATE
    ]

    write_reviews_csv(train_output_path, train_reviews)
    write_reviews_csv(dev_output_path, dev_reviews)

    train_reviews = [
        review
        for review in unique_reviews
        if review["date"] < CUTOFF_DATE
    ]

    dev_reviews = [
        review
        for review in unique_reviews
        if review["date"] >= CUTOFF_DATE
    ]

    total_tokens_processed = sum(
        count_simple_tokens(review["text"])
        for review in unique_reviews
    )

    train_tokens_processed = sum(
        count_simple_tokens(review["text"])
        for review in train_reviews
    )

    dev_tokens_processed = sum(
        count_simple_tokens(review["text"])
        for review in dev_reviews
    )

    # انتقال آمارهای بخش توزیع به متغیر split_counters برای هماهنگی با ساختار summary
    review_counters["retained"] = len(unique_reviews)

    split_counters["train_reviews"] = len(train_reviews)
    split_counters["dev_reviews"] = len(dev_reviews)
    split_counters["total_tokens_processed"] = total_tokens_processed
    split_counters["train_tokens_processed"] = train_tokens_processed
    split_counters["dev_tokens_processed"] = dev_tokens_processed


    training_catalog = {
        review["business_id"]
        for review in train_reviews
    }

    training_user_counts: Dict[str, int] = defaultdict(int)
    for review in train_reviews:
        training_user_counts[review["user_id"]] += 1

    relevance_by_pair: Dict[tuple[str, str], int] = defaultdict(int)

    for review in dev_reviews:
        pair = (review["user_id"], review["business_id"])

        if review["stars"] >= 4.0:
            relevance_by_pair[pair] = 1
        else:
            relevance_by_pair.setdefault(pair, 0)

    sorted_qrels = sorted(relevance_by_pair.items())

    with qrels_output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
    ) as handle:
        handle.write("user_id\tbusiness_id\trelevance\n")

        for (user_id, business_id), relevance in sorted_qrels:
            handle.write(
                f"{user_id}\t{business_id}\t{relevance}\n"
            )

    dev_users = sorted({
        user_id
        for (user_id, business_id), relevance in relevance_by_pair.items()
        if (
                relevance == 1
                and training_user_counts[user_id] >= 3
                and business_id in training_catalog
        )
    })

    with dev_users_output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
    ) as handle:
        handle.write("user_id\n")
        for user_id in dev_users:
            handle.write(f"{user_id}\n")

    split_counters["qrels_pairs"] = len(sorted_qrels)
    split_counters["selected_dev_users"] = len(dev_users)
    split_counters["training_catalog_businesses"] = len(training_catalog)


def write_summary(
        output_path: Path,
        counters: Dict[str, Dict[str, int]],
        arguments: argparse.Namespace,
) -> None:
    business_total = int(counters["business"].get("total_records", 0))
    business_retained = int(counters["business"].get("retained", 0))
    business_discarded = business_total - business_retained

    review_total = int(counters["review"].get("total_records", 0))
    review_retained = int(counters["review"].get("retained", 0))
    review_discarded = review_total - review_retained

    summary = {
        "phase": "NLP Phase 0",
        "cutoff_date": CUTOFF_DATE.isoformat(),

        "inputs": {
            "business_path": str(arguments.business_path),
            "review_path": str(arguments.review_path),
        },

        "outputs": {
            "businesses": "businesses.jsonl",
            "train_reviews": "train_reviews.csv",
            "dev_reviews": "dev_reviews.csv",
            "dev_qrels": "dev_qrels.tsv",
            "dev_users": "dev_users.csv",
        },

        "raw_record_discard_counts": {
            "business_total_records": business_total,
            "business_retained_records": business_retained,
            "business_discarded_records": business_discarded,

            "review_total_records": review_total,
            "review_retained_records": review_retained,
            "review_discarded_records": review_discarded,

            "business_discard_reasons": {
                key: int(value)
                for key, value in sorted(counters["business"].items())
                if key not in {"total_records", "retained"}
            },

            "review_discard_reasons": {
                key: int(value)
                for key, value in sorted(counters["review"].items())
                if key not in {"total_records", "retained"}
            },
        },

        "row_counts": {
            "train_reviews": int(counters["split"].get("train_reviews", 0)),
            "dev_reviews": int(counters["split"].get("dev_reviews", 0)),
            "qrels_pairs": int(counters["split"].get("qrels_pairs", 0)),
            "selected_dev_users": int(counters["split"].get("selected_dev_users", 0)),
            "training_catalog_businesses": int(
                counters["split"].get("training_catalog_businesses", 0)
            ),
        },

        "total_tokens_processed": int(
            counters["split"].get("total_tokens_processed", 0)
        ),

        "token_counts": {
            "train_tokens_processed": int(
                counters["split"].get("train_tokens_processed", 0)
            ),
            "dev_tokens_processed": int(
                counters["split"].get("dev_tokens_processed", 0)
            ),
        },

        "dataset_stats": {
            "business": dict(sorted(counters["business"].items())),
            "review": dict(sorted(counters["review"].items())),
            "split": dict(sorted(counters["split"].items())),
        },
    }

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def count_simple_tokens(text: str) -> int:
    """
    Count whitespace-separated tokens.

    This is a deterministic approximation for reporting
    total tokens processed in preprocessing.
    """
    if not isinstance(text, str):
        return 0

    return len(text.split())

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Yelp Phase 0 preprocessing"
    )

    parser.add_argument(
        "--business-path",
        type=Path,
        required=True,
        help="Path to raw business.json",
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        required=True,
        help="Path to raw review.json",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output workspace directory",
    )

    arguments = parser.parse_args()
    arguments.outdir.mkdir(parents=True, exist_ok=True)

    counters = {
        "business": defaultdict(int),
        "review": defaultdict(int),
        "split": defaultdict(int)
    }

    retained_business_ids = prepare_businesses(
        business_path=arguments.business_path,
        output_path=arguments.outdir / "businesses.jsonl",
        counters=counters["business"],
    )

    prepare_reviews(
        review_path=arguments.review_path,
        retained_business_ids=retained_business_ids,
        train_output_path=arguments.outdir / "train_reviews.csv",
        dev_output_path=arguments.outdir / "dev_reviews.csv",
        qrels_output_path=arguments.outdir / "dev_qrels.tsv",
        dev_users_output_path=arguments.outdir / "dev_users.csv",
        review_counters=counters["review"],
        split_counters=counters["split"],
    )

    write_summary(
        output_path=arguments.outdir / "preprocessing_summary.json",
        counters=counters,
        arguments=arguments,
    )

    print("Phase 0 completed successfully.")
    print(f"Outputs written to: {arguments.outdir}")


if __name__ == "__main__":
    main()