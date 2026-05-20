"""
Airbnb NYC Reviews — Natural Language Processing Evolution Pipeline
====================================================================

This pipeline tells the story of how NLP has evolved by solving the same
problem — Airbnb guest review sentiment classification — with progressively more
powerful techniques.  Every era runs on the same dataset and produces the
same output shape so the performance comparison in Step 9 clearly showcases 
how far the field has evolved.

  Step 1   — Data Loading & Label Generation
             Join reviews.csv + listings.csv on listing_id.
             Threshold review_scores_rating > 4.7 → positive label and < 3.5 → negative label.

  Step 2   — Text Preprocessing
             Full pipeline (clean → tokenise → stopwords → lemmatise) for
             Eras 1–3.  Minimal cleaning for BERT and LLM.

  Step 3   — Train / Test Split  (stratified 80 / 20 with undersampling of the majority class)

  Step 4   — Era 1: Statistical  (Bag-of-Words → TF-IDF)
             Words as count / frequency vectors.  Breaking point: sparsity,
             loss of word order, identical vectors for opposite-meaning text.

  Step 5   — Era 2: Semantic  (Word2Vec)
             Dense vectors — words with similar meaning appear in similar
             contexts.  Breaking point: polysemy — static vectors cannot
             capture context-dependent word meaning.

  Step 6   — Era 3: Sequential  (LSTM)
             Hidden state memory fixes short-term forgetting.  Breaking
             point: sequential computation — cannot parallelise on GPUs.

  Step 7   — Era 4: Attention  (BERT — bert-base-uncased)
             Self-attention lets every token attend to every other token
             simultaneously.  Fully parallelisable; trained on internet scale.

  Step 8   — Era 5: Foundation Models  (LLM — zero-shot & few-shot)
             Scaling unlocks emergent abilities.  No task-specific training
             required; five examples outperform hours of LSTM fine-tuning.

  Step 9   — Era Comparison Table  (accuracy + F1 across all eras)

  Step 10  — Embedding Drift Detection
             Evidently AI on PCA-reduced BERT [CLS] embeddings.

Modularity note
---------------
Each NLP concern lives in src/unstruc/text/.  This file is pure orchestration:
it calls modules in order, passes outputs between them, and logs results to
W&B via ExperimentTracker.  No business logic lives here.
"""

import argparse
from pathlib import Path

import rarfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch

from src.unstruc.text.preprocessor       import preprocess_series
from src.unstruc.text.vectorizer         import Vectorizer
from src.unstruc.text.sequence_model     import build_vocab, train_lstm, predict_lstm
from src.unstruc.text.transformer_encoder import (BertClassifier, get_tokenizer,
                                                   train_bert, predict_bert,
                                                   extract_embeddings)
from src.unstruc.text.llm_client         import classify_batch
from src.experiment_tracking             import ExperimentTracker
from src.drift_detection_evidently       import (run_embedding_drift_report,
                                                  parse_embedding_drift_results)


# ── Path configuration ────────────────────────────────────────────────────────

REVIEWS_RAR   = "data/raw/airbnb/reviews.rar"
LISTINGS_PATH = "data/raw/airbnb/listings.csv"
MODEL_DIR     = "models/nlp"
PLOTS_DIR     = "outputs/plots"

# ── Sampling configuration — adjust for available compute ────────────────────
#
# SAMPLE_SIZE     controls Eras 1–3 (statistical, semantic, LSTM).
# BERT_TRAIN_SIZE controls how many samples BERT is fine-tuned on.
#                 2 000 samples × 2 epochs ≈ 5–10 min on CPU, ~1 min on GPU.
# LLM_EVAL_SIZE   controls API call count — keep small to manage cost.

POSITIVE_THRESHOLD  = 4.7   # listing score ≥ this → positive (clearly good)
NEGATIVE_THRESHOLD  = 3.5   # listing score ≤ this → negative (clearly bad)
MAX_POSITIVE_SAMPLES = 2_000 # undersample majority class; keep all negatives
# Reviews from listings scoring between 3.5–4.7 are dropped — ambiguous signal.
BERT_TRAIN_SIZE = 200
LLM_EVAL_SIZE   = 100
RANDOM_STATE    = 42

# ── W&B configuration ─────────────────────────────────────────────────────────

WANDB_PROJECT = "dsma-lecture6-nlp"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_rar(rar_path: str) -> str:
    """
    Extract the CSV from a RAR archive and return the path to the extracted file.
    Skips extraction if the CSV already exists on disk.
    """
    rar_path = Path(rar_path)
    with rarfile.RarFile(rar_path) as rf:
        csv_names = [n for n in rf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError(f"No CSV found inside {rar_path}")
        csv_name    = csv_names[0]
        extract_dir = rar_path.parent
        extracted   = extract_dir / csv_name
        if not extracted.exists():
            print(f"  Extracting {csv_name} from {rar_path.name} ...")
            rf.extract(csv_name, path=str(extract_dir))
        else:
            print(f"  Found extracted file, skipping extraction: {extracted.name}")
    return str(extracted)


def _print_header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _eval(y_true, y_pred, name: str) -> dict:
    acc          = accuracy_score(y_true, y_pred)
    f1           = f1_score(y_true, y_pred, average="binary")
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    tnr          = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  {name:<40}  Accuracy: {acc:.4f}   F1: {f1:.4f}   TNR: {tnr:.4f}")
    return {"name": name, "accuracy": acc, "f1": f1, "tnr": tnr}


def _era_comparison_table(results: list[dict]):
    print(f"\n  {'Era / Method':<40} {'Accuracy':>10} {'F1 Score':>10} {'TNR':>8}")
    print("  " + "-" * 72)
    for r in results:
        print(f"  {r['name']:<40} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['tnr']:>8.4f}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(wandb_project: str = WANDB_PROJECT):
    era_results = []

    # ── Step 1: Data Loading & Label Generation ────────────────────────────────
    _print_header("STEP 1 — Data Loading & Label Generation")

    reviews_csv = _extract_rar(REVIEWS_RAR)
    reviews_df  = pd.read_csv(reviews_csv)
    listings_df = pd.read_csv(
        LISTINGS_PATH,
        usecols     = ["id", "review_scores_rating"],
        low_memory  = False,
    )

    # Normalise rating scale: older Inside Airbnb exports use 0–100, newer use 0–5.
    if listings_df["review_scores_rating"].max() > 5:
        listings_df["review_scores_rating"] = listings_df["review_scores_rating"] / 20.0

    listings_slim = listings_df.rename(columns={"id": "listing_id"})
    df = reviews_df.merge(listings_slim, on="listing_id", how="inner")
    df = df.dropna(subset=["comments", "review_scores_rating"]).reset_index(drop=True)

    # Confidence-based labelling: drop the ambiguous middle band (3.5–4.7).
    # Keeps only reviews whose listing score is clearly positive or clearly
    # negative, giving every era a cleaner learning signal.
    df_pos = df[df["review_scores_rating"] >= POSITIVE_THRESHOLD].copy()
    df_neg = df[df["review_scores_rating"] <= NEGATIVE_THRESHOLD].copy()
    df_pos["label"] = 1
    df_neg["label"] = 0

    # Undersample the majority (positive) class; keep all negatives.
    df_pos = df_pos.sample(
        min(MAX_POSITIVE_SAMPLES, len(df_pos)), random_state=RANDOM_STATE
    )
    df = (
        pd.concat([df_pos, df_neg], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    pos_pct = df["label"].mean() * 100
    print(f"  Positive   : {df['label'].sum():,} reviews  (score ≥ {POSITIVE_THRESHOLD}, capped at {MAX_POSITIVE_SAMPLES:,})")
    print(f"  Negative   : {(df['label'] == 0).sum():,} reviews  (score ≤ {NEGATIVE_THRESHOLD}, all kept)")
    print(f"  Total      : {len(df):,}   class split: {pos_pct:.1f}% / {100 - pos_pct:.1f}%")

    # ── Step 2: Text Preprocessing ─────────────────────────────────────────────
    _print_header("STEP 2 — Text Preprocessing")

    print("  Full preprocessing (Eras 1–3): clean → tokenise → stopwords → lemmatise ...")
    df["text_full"] = preprocess_series(df["comments"], mode="full")

    print("  Minimal preprocessing (BERT / LLM): clean only — model handles its own tokenisation ...")
    df["text_bert"] = preprocess_series(df["comments"], mode="minimal")

    # Drop rows where full preprocessing produced an empty string — this happens
    # when a review consists entirely of stopwords or non-alphabetic characters.
    # Filtering here keeps all downstream steps, including the LLM API, free of
    # empty-content errors.
    before = len(df)
    df = df[df["text_full"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} reviews with empty text after preprocessing.")

    print(f"\n  Sample raw     : {df['comments'].iloc[0][:120]}")
    print(f"  Sample full    : {df['text_full'].iloc[0][:120]}")
    print(f"  Sample minimal : {df['text_bert'].iloc[0][:120]}")

    # ── Step 3: Train / Test Split ─────────────────────────────────────────────
    _print_header("STEP 3 — Train / Test Split (stratified 80 / 20)")

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        df["text_full"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    X_train_bert, X_test_bert, _, _ = train_test_split(
        df["text_bert"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    print(f"  Train : {len(X_train_full):,} reviews")
    print(f"  Test  : {len(X_test_full):,} reviews")

    # ══════════════════════════════════════════════════════════════════════════
    # ERA 1 — STATISTICAL (Bag-of-Words & TF-IDF)
    #
    # Represent each review as a sparse vector of word counts (BoW) or
    # normalised frequencies (TF-IDF).  Train a logistic regression on top.
    #
    # Demonstration of the breaking point:
    #   - "This place is absolutely amazing" vs
    #     "This place is absolutely not amazing"
    #   Both produce nearly identical vectors; the model cannot distinguish them.
    #   Print the sparsity of the BoW matrix to make the issue tangible.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 4 — Era 1: Statistical (BoW & TF-IDF)")

    for method, label in [("bow",   "BoW   + Logistic Regression"),
                           ("tfidf", "TF-IDF + Logistic Regression")]:
        vec          = Vectorizer(method=method, max_features=10_000)
        X_train_vec  = vec.fit_transform(X_train_full)
        X_test_vec   = vec.transform(X_test_full)

        if method == "bow":
            sparsity = 1.0 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1])
            print(f"  BoW matrix shape : {X_train_vec.shape}   sparsity: {sparsity:.2%}")

        clf = LogisticRegression(max_iter=1_000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
        clf.fit(X_train_vec, y_train)
        era_results.append(_eval(y_test, clf.predict(X_test_vec), label))

    # ══════════════════════════════════════════════════════════════════════════
    # ERA 2 — SEMANTIC (Word2Vec)
    #
    # Train Word2Vec on the corpus and represent each review by the mean of
    # its word vectors.  Dense, low-dimensional (100-d) — no sparsity problem.
    #
    # Classic vector arithmetic to motivate the approach:
    #   king − man + woman ≈ queen
    #
    # Demonstration of the breaking point: polysemy.
    #   "The host had a great character" vs
    #   "The neighbourhood had a dangerous character"
    #   Word2Vec assigns a single static vector to 'character' — it cannot
    #   know which sense is meant without reading the whole sentence in context.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 5 — Era 2: Semantic (Word2Vec)")

    w2v_vec     = Vectorizer(method="word2vec", vector_size=100)
    X_train_w2v = w2v_vec.fit_transform(X_train_full)
    X_test_w2v  = w2v_vec.transform(X_test_full)
    print(f"  Word2Vec vocab size : {w2v_vec.vocab_size:,}")
    print(f"  Embedding dimension : {X_train_w2v.shape[1]}")

    clf_w2v = LogisticRegression(max_iter=1_000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
    clf_w2v.fit(X_train_w2v, y_train)
    era_results.append(_eval(y_test, clf_w2v.predict(X_test_w2v),
                             "Word2Vec + Logistic Regression"))

    # ══════════════════════════════════════════════════════════════════════════
    # ERA 3 — SEQUENTIAL (LSTM)
    #
    # LSTMs read text left-to-right, maintaining a hidden state that compresses
    # everything seen so far.  Gating mechanisms (forget, input, output gates)
    # prevent the vanishing gradient problem that plagued vanilla RNNs, enabling
    # the model to learn dependencies over hundreds of tokens.
    #
    # Demonstration of the breaking point: try a 400-word review and observe
    # how training time grows.  Token N cannot be computed until token N-1 is
    # done — this sequential dependency prevents GPU parallelisation and is
    # the direct motivation for the Transformer.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 6 — Era 3: Sequential (LSTM)")

    vocab = build_vocab(X_train_full, max_vocab=20_000)
    print(f"  Vocabulary size : {len(vocab):,}")

    lstm_model = train_lstm(
        X_train_texts = X_train_full.tolist(),
        y_train       = y_train.tolist(),
        vocab         = vocab,
        epochs        = 50,
        batch_size    = 64,
    )
    era_results.append(_eval(y_test, predict_lstm(lstm_model, X_test_full.tolist(), vocab),
                             "LSTM"))

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    lstm_path = Path(MODEL_DIR) / "lstm.pt"
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"  LSTM weights saved → {lstm_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # ERA 4 — ATTENTION (BERT)
    #
    # Transformers discard recurrence entirely.  The self-attention mechanism
    # lets every token directly attend to every other token in the sequence,
    # computing a weighted context score for each pair simultaneously.
    # Fully parallelisable → can be trained on internet-scale data.
    #
    # bert-base-uncased is pre-trained on BookCorpus + English Wikipedia.
    # We add a single linear layer on top of the [CLS] pooled representation
    # and fine-tune for 2 epochs on a subset of the training data.
    #
    # Note the preprocessing shift: we pass text_bert (minimal cleaning) rather
    # than text_full.  Lemmatisation and stopword removal would destroy the
    # subword distributions BERT was pretrained on.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 7 — Era 4: Attention (BERT — bert-base-uncased)")

    rng      = np.random.RandomState(RANDOM_STATE)
    bert_idx = rng.choice(len(X_train_bert), BERT_TRAIN_SIZE, replace=False)
    X_bert_train = X_train_bert.iloc[bert_idx].tolist()
    y_bert_train = y_train.iloc[bert_idx].tolist()

    print(f"  Fine-tuning on {BERT_TRAIN_SIZE:,} samples for 2 epochs ...")
    bert_model, bert_tokenizer = train_bert(
        X_train_texts = X_bert_train,
        y_train       = y_bert_train,
        epochs        = 10,
        batch_size    = 16,
    )
    era_results.append(_eval(y_test, predict_bert(bert_model, bert_tokenizer,
                                                   X_test_bert.tolist()),
                             "BERT (bert-base-uncased)"))

    bert_save_path = Path(MODEL_DIR) / "bert"
    bert_model.bert.save_pretrained(str(bert_save_path))
    bert_tokenizer.save_pretrained(str(bert_save_path))
    print(f"  BERT weights saved → {bert_save_path}/")

    # ══════════════════════════════════════════════════════════════════════════
    # ERA 5 — FOUNDATION MODELS (LLM)
    #
    # Scaling transformers to billions of parameters and training on internet-
    # scale corpora unlocks emergent abilities: reasoning, translation, coding —
    # all from predicting the next token.  No task-specific training required
    # for zero-shot; three in-context examples (few-shot) often match or exceed
    # a fine-tuned BERT trained on thousands of labelled samples.
    #
    # Cost note: each call makes one API request.  LLM_EVAL_SIZE controls
    # the total number of API calls.  Default is 100 samples (~$0.002).
    # ══════════════════════════════════════════════════════════════════════════

    # _print_header("STEP 8 — Era 5: Foundation Models (LLM — zero-shot & few-shot)")

    # # LLMs understand natural language — use minimally cleaned text, not the
    # # lemmatized/stopword-removed version sent to statistical and LSTM models.
    # llm_idx = rng.choice(len(X_test_bert), LLM_EVAL_SIZE, replace=False)
    # X_llm   = X_test_bert.iloc[llm_idx].tolist()
    # y_llm   = y_test.iloc[llm_idx].tolist()

    # # Few-shot examples: 2 positive + 1 negative from training set
    # pos_texts = X_train_bert[y_train == 1].iloc[:5].tolist()
    # neg_texts = X_train_bert[y_train == 0].iloc[:5].tolist()
    # few_shot_examples = (
    #     [{"text": t, "label": 1} for t in pos_texts] +
    #     [{"text": t, "label": 0} for t in neg_texts]
    # )

    # print(f"  Classifying {LLM_EVAL_SIZE} samples  (zero-shot) ...")
    # y_pred_zs = classify_batch(X_llm, mode="zero_shot")
    # era_results.append(_eval(y_llm, y_pred_zs, "LLM — zero-shot"))

    # print(f"  Classifying {LLM_EVAL_SIZE} samples  (few-shot, 3 examples) ...")
    # y_pred_fs = classify_batch(X_llm, mode="few_shot", examples=few_shot_examples)
    # era_results.append(_eval(y_llm, y_pred_fs, "LLM — few-shot (3 examples)"))

    # ── Step 9: Era Comparison Table ────────────────────────────────────────────
    _print_header("STEP 9 — Era Comparison: The NLP Evolution")
    _era_comparison_table(era_results)

    tracker = ExperimentTracker(
        project  = wandb_project,
        run_name = "nlp-era-comparison",
        tags     = ["nlp", "airbnb", "era-comparison"],
        config   = {
            "max_positive_samples": MAX_POSITIVE_SAMPLES,
            "bert_train_size":      BERT_TRAIN_SIZE,
            "llm_eval_size":        LLM_EVAL_SIZE,
            "positive_threshold":   POSITIVE_THRESHOLD,
            "negative_threshold":   NEGATIVE_THRESHOLD,
        },
    )
    summary = {}
    for r in era_results:
        key = r["name"].lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        summary[f"{key}_accuracy"] = r["accuracy"]
        summary[f"{key}_f1"]       = r["f1"]
        summary[f"{key}_tnr"]      = r["tnr"]
    tracker.log_summary(summary)
    url = tracker.finish()
    print(f"\n  W&B run logged → {url}")

    # ── Step 10: Embedding Drift Detection ─────────────────────────────────────
    _print_header("STEP 10 — Embedding Drift Detection (Evidently AI)")

    # Simulate a reference → current distribution shift by comparing the first
    # and second halves of the test set.  In production, reference would be
    # the training-time distribution and current would be live inference data.
    mid   = len(X_test_bert) // 2
    X_ref = X_test_bert.iloc[:mid].tolist()
    X_cur = X_test_bert.iloc[mid:].tolist()

    print(f"  Extracting BERT [CLS] embeddings  (reference: {len(X_ref)}, current: {len(X_cur)}) ...")
    ref_emb = extract_embeddings(bert_model, bert_tokenizer, X_ref)
    cur_emb = extract_embeddings(bert_model, bert_tokenizer, X_cur)

    drift_report   = run_embedding_drift_report(ref_emb, cur_emb, n_components=20)
    drift_results  = parse_embedding_drift_results(drift_report)

    print(f"\n  Embedding drift detected : {drift_results['overall_drift']}")
    print(f"  PCA components drifted   : {drift_results['n_drifted']} "
          f"({drift_results['share_drifted']:.1%} of 20 components)")

    Path("outputs").mkdir(exist_ok=True)
    drift_html = Path("outputs") / "embedding_drift_report.html"
    drift_report.save_html(str(drift_html))
    print(f"  Embedding drift report   → {drift_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help="W&B project name to log runs into")
    args = parser.parse_args()
    run_pipeline(wandb_project=args.wandb_project)
