import time
from collections import Counter, defaultdict
from typing import List, Tuple
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
import json
from pathlib import Path
from transformer_model import get_transformer_scores, TransformerSentimentScorer

# تعریف جنبه‌های قانونی (Canonical Aspects) به ترتیب اولویت تعیین شده برای حل تساوی
CANONICAL_ASPECTS = ["food", "service", "ambiance", "price", "location"]
THETA = 0.35
EPS = 1e-9


def initialize_nlp_and_seeds(model_name: str = "en_core_web_md") -> Tuple[
    spacy.language.Language, List[spacy.tokens.Token]]:
    """
    لود کردن مدل spaCy و آماده‌سازی توکن‌های Seed برای مقایسه بردارها.
    """
    nlp = spacy.load(model_name, disable=["ner"])
    seed_docs = [nlp(aspect) for aspect in CANONICAL_ASPECTS]
    seed_tokens = [doc[0] for doc in seed_docs]
    return nlp, seed_tokens


def expand_opinion_phrase(adj_token: spacy.tokens.Token) -> str:
    """
    ساخت عبارت نظر با الحاق صفت به همراه قیدهای مقدار (advmod) و منفی‌کننده‌های (neg) متصل به آن.
    مرتب‌سازی بر اساس ایندکس توکن‌ها در جمله انجام می‌شود.
    """
    selected_tokens = [adj_token]
    for child in adj_token.children:
        if child.dep_ in ("advmod", "neg"):
            selected_tokens.append(child)

    # مرتب‌سازی توکن‌ها بر اساس ترتیب حضور در متن اصلی
    selected_tokens = sorted(selected_tokens, key=lambda t: t.i)
    return " ".join([t.text for t in selected_tokens])


def process_train_dataset_with_phase2(
        csv_path: str,
        nlp: spacy.language.Language,
        seed_tokens: List[spacy.tokens.Token],
        n_process: int = 4,
        use_transformer: bool = False,
        transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    پردازش یکپارچه نقدها (فاز ۱ و ۲)، استخراج احساسات مبتنی بر درخت نحو و ساخت ماتریس‌های U و V.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("")

    # اطمینان از وجود ستون‌های مورد نیاز
    assert "user_id" in df.columns and "business_id" in df.columns and "text" in df.columns

    reviews_list = df["text"].tolist()
    #-----------------------------------
    # مقداردهی اولیه تحلیل‌گر احساسات
    # For transformer
    # ----------------------------------
    sia = SentimentIntensityAnalyzer()
    transformer_scorer = None

    if use_transformer:
        print(f"Initializing Transformer model: {transformer_model}...")
        # ساخت یک نمونه از کلاس ترنسفورمر به صورت سراسری برای این تابع
        transformer_scorer = TransformerSentimentScorer(
            model_name=transformer_model,
            batch_size=16
        )

    # ساختارهای نگهداری نتایج تجمعی
    user_aspect_counts = defaultdict(lambda: Counter())
    # ساختار ذخیره‌سازی s(r, a) برای هر آیتم: { item_id: { aspect: [s_ra_1, s_ra_2, ...] } }
    item_aspect_scores = defaultdict(lambda: defaultdict(list))
    global_aspect_frequencies = Counter()

    # ستون‌های جدید در دیتابیس برای فرکانس و امتیاز احساسات هر نقد
    for aspect in CANONICAL_ASPECTS:
        df[f"aspect_{aspect}"] = 0
        df[f"sent_{aspect}"] = None  # مقدار پیش‌فرض None (عدم ذکر جنبه در نقد)

    print(f"Processing {len(reviews_list)} reviews using {n_process} processes...")
    start_time = time.time()

    # پردازش موازی و یکپارچه نقدها با استفاده از nlp.pipe
    docs = nlp.pipe(reviews_list, n_process=n_process, batch_size=1000)

    for idx, doc in enumerate(docs):
        user = df.iloc[idx]["user_id"]
        item = df.iloc[idx]["business_id"]

        # برای هر نقد، ابتدا کاندیداها و ساختارهای مربوط به آن‌ها را استخراج می‌کنیم
        # ساختاری برای گروه‌بندی عبارات احساسی استخراج‌شده بر اساس جنبه نگاشت‌شده در سطح این نقد
        review_aspect_phrases = defaultdict(list)
        review_aspect_mentions = []

        for chunk in doc.noun_chunks:
            root_token = chunk.root

            # فیلترهای اولیه بر اساس مشخصات فاز ۱
            if (
                    root_token.pos_ == "PRON" or
                    root_token.is_stop or
                    root_token.is_punct or
                    not root_token.has_vector
            ):
                continue

            # پیدا کردن نزدیک‌ترین جنبه بر اساس Cosine Similarity با رعایت ترتیب اولویت در تساوی
            best_aspect = None
            max_similarity = -1.0
            for aspect_idx, seed_token in enumerate(seed_tokens):
                sim = root_token.similarity(seed_token)
                if sim > max_similarity:
                    max_similarity = sim
                    best_aspect = CANONICAL_ASPECTS[aspect_idx]

            # بررسی آستانه شباهت
            if best_aspect is not None and max_similarity >= THETA:
                review_aspect_mentions.append(best_aspect)

                # فاز ۲: استخراج توصیف‌گرهای احساسی (Opinion Modifiers)
                phrases = []

                # الگو ۱: صفت‌های amod که فرزند مستقیم ریشه هستند
                for child in root_token.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        phrases.append(expand_opinion_phrase(child))

                # الگو ۲: ساختار رابط (Copular Verb)
                if root_token.dep_ in ("nsubj", "nsubjpass") and root_token.head.pos_ in ("VERB", "AUX"):
                    verb = root_token.head
                    for child in verb.children:
                        if child.dep_ == "acomp" and child.pos_ == "ADJ":
                            phrases.append(expand_opinion_phrase(child))

                # ذخیره عبارات استخراج‌شده برای این Mention خاص
                review_aspect_phrases[best_aspect].extend(phrases)

        # ثبت فرکانس ذکر جنبه‌ها در سطح نقد و کاربر
        review_counter = Counter(review_aspect_mentions)

        for aspect in CANONICAL_ASPECTS:
            count_val = int(review_counter[aspect])

            df.at[idx, f"aspect_{aspect}"] = count_val
            user_aspect_counts[user][aspect] += count_val
            global_aspect_frequencies[aspect] += count_val

        # محاسبه s(r, a) برای هر جنبه ذکر شده در این نقد
        for aspect in set(review_aspect_mentions):
            phrases = review_aspect_phrases[aspect]
            if phrases:
                if use_transformer:
                    # استفاده از متد کلاس جهت استخراج دسته‌ای امتیازات بدون لود مجدد مدل
                    scores = transformer_scorer.score_texts(phrases)
                else:
                    scores = [sia.polarity_scores(p)["compound"] for p in phrases]
                s_ra = sum(scores) / len(scores)
            else:
                # ذکر شده اما مودیفایری پیدا نشده است
                s_ra = 0.0

            df.at[idx, f"sent_{aspect}"] = s_ra
            # افزودن امتیاز این نقد به لیست نمره‌های آیتم جهت محاسبه ماتریس V
            item_aspect_scores[item][aspect].append(s_ra)

    processing_seconds = time.time() - start_time
    print(f"Processing completed in {processing_seconds:.2f} seconds.")
    df.to_csv("HW3_data/train_df_with_aspects.csv", index=False)

    # ساخت ماتریس توجه کاربر U (User-Aspect Attention Matrix)
    print("Building User-Aspect Attention Matrix (U)...")
    users = sorted(df["user_id"].unique())
    U = pd.DataFrame(0.0, index=users, columns=CANONICAL_ASPECTS, dtype=float)

    for user in users:
        counts = user_aspect_counts[user]
        total_mentions = sum(counts[a] for a in CANONICAL_ASPECTS)
        denom = max(total_mentions, EPS)
        for aspect in CANONICAL_ASPECTS:
            # اگر کاربر هیچ جنبه‌ای را ذکر نکرده باشد، کل سطر صفر خواهد بود
            U.loc[user, aspect] = counts[aspect] / denom if total_mentions > 0 else 0.0

    U.to_csv("HW3_data/user_attention_matrix_U.csv")


    # ساخت ماتریس کیفیت آیتم V (Item-Aspect Quality Matrix)
    print("Building Item-Aspect Quality Matrix (V)...")
    items = sorted(df["business_id"].unique())
    V = pd.DataFrame(0.0, index=items, columns=CANONICAL_ASPECTS, dtype=float)

    for item in items:
        for aspect in CANONICAL_ASPECTS:
            scores = item_aspect_scores[item][aspect]
            count_i_a = len(scores)  # تعداد نقدهای این آیتم که این جنبه را ذکر کرده‌اند
            denom = max(count_i_a, EPS)
            # اگر آیتم هیچ نقدی با ذکر این جنبه نداشته باشد، مقدار ۰ ثبت می‌شود
            V.loc[item, aspect] = sum(scores) / denom if count_i_a > 0 else 0.0

    V.to_csv("HW3_data/item_quality_matrix_V.csv")

    save_aspect_summary(
        output_path="HW3_data/aspect_summary.json",
        aspect_frequencies=global_aspect_frequencies,
        processing_seconds=processing_seconds,
        model_name=nlp.meta.get("name", "unknown_spacy_model"),
        n_reviews=len(df),
        n_process=n_process,
    )

    return df, U, V

def save_aspect_summary(
        output_path: str,
        aspect_frequencies: Counter,
        processing_seconds: float,
        model_name: str,
        n_reviews: int,
        n_process: int,
) -> None:
    """
    Save aspect extraction summary for final Assignment summary.json.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase": "Aspect Extraction and VADER Sentiment",
        "model_name": model_name,
        "n_reviews_processed": int(n_reviews),
        "n_process": int(n_process),
        "processing_seconds": float(processing_seconds),
        "aspect_frequencies": {
            aspect: int(aspect_frequencies.get(aspect, 0))
            for aspect in CANONICAL_ASPECTS
        },
        "parameters": {
            "theta": float(THETA),
            "eps": float(EPS),
            "canonical_aspects": CANONICAL_ASPECTS,
        },
        "package_model_versions": {
            "pandas": pd.__version__,
            "spacy": spacy.__version__,
        },
    }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved aspect summary to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy-model", type=str, default="en_core_web_md")
    parser.add_argument("--n-process", type=int, default=4)
    parser.add_argument("--use-transformer", action="store_true")
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english"
    )

    args = parser.parse_args()

    nlp, seed_tokens = initialize_nlp_and_seeds(args.spacy_model)

    train_df, U, V = process_train_dataset_with_phase2(
        csv_path="HW3_data/train_reviews.csv",
        nlp=nlp,
        seed_tokens=seed_tokens,
        n_process=args.n_process,
        use_transformer=args.use_transformer,
        transformer_model=args.transformer_model,
    )

    print("\nSample processed reviews:")
    print(train_df[["text", "aspect_food", "sent_food", "aspect_service", "sent_service"]].head())

    # ذخیره خروجی‌ها
    train_df.to_csv("HW3_data/train_df_with_aspects.csv", index=False)
    U.to_csv("HW3_data/user_attention_matrix_U.csv")
    V.to_csv("HW3_data/item_quality_matrix_V.csv")
    print("\nSaved output files successfully!")