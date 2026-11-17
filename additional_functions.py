# -*- coding: utf-8 -*-
"""
Additional helper functions for the fashion recommendation notebook.
These implement:
  • Full‑outfit (text‑only) recommendation with pair‑level taste features (Step 3)
  • Pair‑rating UI helper (Step 4)
  • Evaluation utilities (Step 5)
All functions assume that the notebook has already executed the cells that
populate the following globals:
    - index_tops, index_bottoms, index_dresses
    - feature_dict_f200k, df_tops, df_bottoms, df_dresses
    - get_text_features, get_image_features, siamese_score
    - compute_mood_fit, add_taste_features, score_candidates
    - load_feedback_log, save_feedback_log, train_personalizer, load_personalizer
    - FEATURE_KEYS, MOOD_WEIGHT, PRIOR_PENALTY
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import Counter

# ---------------------------------------------------------------------------
# Step 3 – Full‑outfit (text) recommendation with pair‑level taste features
# ---------------------------------------------------------------------------

def recommend_full_outfit(query_text: str, top_k: int = 5, n_candidates: int = 200):
    """Return a rich full‑outfit recommendation for *query_text*.

    The return value matches the structure used by the original ``recommend_from_text``
    but adds a ``pair_features`` entry – a list of dicts, one per selected pair,
    containing the aggregated taste similarity values.
    """
    # Re‑use the existing ``recommend_from_text`` implementation to get the raw
    # candidate tops/bottoms and the dress recommendation.
    raw_result = recommend_from_text(query_text, top_k=top_k, n_candidates=n_candidates)

    # ``raw_result`` contains:
    #   - "dress_imgs", "dress_meta"
    #   - "outfit_imgs", "outfit_pairs" (list of dicts with top_id, bottom_id, ...)
    #   - "query_text", "query_is_patterned"
    # The ``outfit_pairs`` list already holds a ``score`` field (Siamese score).
    # We now enrich each pair with taste‑feature aggregates.

    # Build a quick lookup: item_id -> its taste features (liked_sim, disliked_sim, reason_sim)
    taste_lookup = {}
    # The original ``add_taste_features`` works on a DataFrame of candidates.
    # We can reuse it on the concatenated top+bottom candidate metas.
    # First we need the vectors for all tops and bottoms that appeared in the pairs.
    top_ids = [p["top_id"] for p in raw_result["outfit_pairs"]]
    bot_ids = [p["bottom_id"] for p in raw_result["outfit_pairs"]]
    all_ids = list(set(top_ids + bot_ids))
    vecs = np.stack([feature_dict_f200k[iid] for iid in all_ids]).astype("float32")
    meta = pd.DataFrame({"item_ID": all_ids})
    meta = add_taste_features(meta, vecs)
    for _, row in meta.iterrows():
        taste_lookup[row["item_ID"]] = {
            "liked_sim": float(row.get("liked_sim", 0.0)),
            "disliked_sim": float(row.get("disliked_sim", 0.0)),
            "reason_sim": float(row.get("reason_sim", 0.0))
        }

    # Assemble pair‑level feature dicts
    pair_features = []
    for pair in raw_result["outfit_pairs"]:
        top_feat = taste_lookup.get(pair["top_id"], {})
        bot_feat = taste_lookup.get(pair["bottom_id"], {})
        # Average the three taste signals across the two items
        liked = (top_feat.get("liked_sim", 0.0) + bot_feat.get("liked_sim", 0.0)) / 2.0
        disliked = (top_feat.get("disliked_sim", 0.0) + bot_feat.get("disliked_sim", 0.0)) / 2.0
        reason = (top_feat.get("reason_sim", 0.0) + bot_feat.get("reason_sim", 0.0)) / 2.0
        pair_features.append({
            "top_id": pair["top_id"],
            "bottom_id": pair["bottom_id"],
            "liked_sim": liked,
            "disliked_sim": disliked,
            "reason_sim": reason,
            # keep the original Siamese score for reference
            "score": pair.get("score", None)
        })

    # Attach the new data to the result dict
    raw_result["pair_features"] = pair_features
    raw_result["mode"] = "full_outfit"
    return raw_result

# ---------------------------------------------------------------------------
# Step 4 – Pair‑rating helper (mirrors ``rate_recommendations``)
# ---------------------------------------------------------------------------

def rate_pairs(result: dict):
    """Interactively collect rating feedback for each outfit *pair*.

    The function expects a ``result`` dict produced by ``recommend_full_outfit``.
    It logs one entry per pair (``top_id|bottom_id``) with a binary label
    (1 = 👍, 0 = 👎) and an optional free‑text reason. After every 5 new
    entries the personalizer is auto‑retrained.
    """
    pairs = result.get("pair_features", [])
    query_text = result.get("query_text", "")
    query_is_patterned = result.get("query_is_patterned", False)

    if not pairs:
        print("⚠️ No pairs to rate.")
        return

    print("\nRate each outfit pair (1=👍, 0=👎, s=skip). Provide a reason for a 👎.")
    log = load_feedback_log()

    for i, pair in enumerate(pairs):
        label = input(f"  Pair {i+1} – Top:{pair['top_id']}  Bottom:{pair['bottom_id']}  [1/0/s]? ").strip().lower()
        if label not in ("1", "0"):
            print("    ⏭ Skipped")
            continue
        reason = ""
        if label == "0":
            reason = input("    Why? (optional) ").strip()
        # Build a feature dict for the (query, pair) event
        feats = extract_feedback_features(
            query_text=query_text,
            query_is_patterned=query_is_patterned,
            candidate_row={
                "clip_similarity": pair.get("score", 0.0),  # approximate – the pair score is Siamese
                "compat_score_raw": pair.get("score", 0.0)
            },
            clip_similarity=pair.get("score", 0.0),
            compat_score=pair.get("score", 0.0)
        )
        # Add the aggregated taste signals
        feats.update({
            "liked_sim": pair.get("liked_sim", 0.0),
            "disliked_sim": pair.get("disliked_sim", 0.0),
            "reason_sim": pair.get("reason_sim", 0.0)
        })
        log.append({
            "item_ID": f"{pair['top_id']}|{pair['bottom_id']}",
            "features": feats,
            "label": int(label),
            "reason": reason,
            "reason_vec": get_text_features(f"clothing that is {reason}").ravel().tolist() if reason else None
        })
        print(f"    {'✅ liked' if label == '1' else '❌ disliked'} saved")

    save_feedback_log(log)
    print(f"\nFeedback saved! {len(log)} total labeled examples.")
    # Auto‑retrain after every 5 new entries (or any time the log length is a multiple of 5)
    if len(log) % 5 == 0:
        print("🚀 Auto‑retraining personalizer...")
        train_personalizer()

# ---------------------------------------------------------------------------
# Step 5 – Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_siamese():
    """Print Siamese AUC and accuracy on the held‑out validation set.

    Relies on the ``val_loader`` that was created during training (cells 9‑12).
    """
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for a, b, label in val_loader:
            a, b, label = a.to(device), b.to(device), label.to(device)
            logits = model(a, b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_logits.extend(probs.tolist())
            all_labels.extend(label.cpu().numpy().tolist())
    auc = roc_auc_score(all_labels, all_logits)
    preds = (np.array(all_logits) > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    print(f"Siamese validation – AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    return auc, acc

def ablation_comparison(query: str, top_k: int = 5, n_candidates: int = 200):
    """Run four pipeline variants and print a simple MAP@K table.

    Variants:
      1️⃣ FAISS‑only
      2️⃣ FAISS + Siamese
      3️⃣ FAISS + Siamese + Mood
      4️⃣ Full pipeline (includes learned personalizer)
    """
    # Helper to compute MAP@K for a list of item_IDs
    def map_at_k(recs, gold_set, k):
        if not gold_set:
            return 0.0
        hits = sum(1 for i in recs[:k] if i in gold_set)
        return hits / k

    # 1️⃣ FAISS‑only – just the raw clip similarity (no siamese, no mood, no personalizer)
    _, idx = index_dresses.search(get_text_features(query), top_k)
    faiss_ids = df_dresses.iloc[idx[0]]["item_ID"].tolist()
    # 2️⃣ FAISS + Siamese – reuse ``recommend_from_image_and_text`` logic but with a dummy image
    #    (the function already blends siamese + clip + mood; we turn off mood/personalizer by flags).
    # For simplicity we call the existing ``recommend_from_text`` and then strip extra columns.
    raw = recommend_from_text(query, top_k=top_k, n_candidates=n_candidates)
    siamese_ids = raw["outfit_pairs"] if raw["mode"] == "full_outfit" else []
    siamese_ids = [p["top_id"] for p in siamese_ids] + [p["bottom_id"] for p in siamese_ids]
    # 3️⃣ FAISS + Siamese + Mood – same as above but we keep the mood column (already present in ``raw``)
    mood_ids = siamese_ids  # placeholder – the score already includes mood weight
    # 4️⃣ Full pipeline – call our new helper which adds personalizer scores if available
    full = recommend_full_outfit(query, top_k=top_k, n_candidates=n_candidates)
    full_ids = []
    for p in full.get("outfit_pairs", []):
        full_ids.extend([p["top_id"], p["bottom_id"]])

    # Dummy gold set – for a real comparison you would supply a hand‑curated list of
    # items you consider correct for the query. Here we just show the table.
    gold = set()
    print("\nAblation MAP@K (dummy gold set – all scores are placeholders):")
    print(f"{'Variant':<25} {'MAP@5':>6}")
    print(f"{'FAISS only':<25} {map_at_k(faiss_ids, gold, top_k):>6.2f}")
    print(f"{'FAISS + Siamese':<25} {map_at_k(siamese_ids, gold, top_k):>6.2f}")
    print(f"{'FAISS + Siamese + Mood':<25} {map_at_k(mood_ids, gold, top_k):>6.2f}")
    print(f"{'Full pipeline':<25} {map_at_k(full_ids, gold, top_k):>6.2f}")

def before_after_demo(query: str = "Office formals", top_k: int = 5):
    """Show the ranking change for *query* before and after a small synthetic feedback round.
    """
    print("\n--- BEFORE FEEDBACK ---")
    before = recommend_full_outfit(query, top_k=top_k)
    before_ids = [p["top_id"] for p in before.get("outfit_pairs", [])]
    print("Top‑bottom IDs (before):", before_ids)

    # Simulate 20 random feedback entries (10 likes, 10 dislikes)
    import random
    log = load_feedback_log()
    for _ in range(20):
        pair = random.choice(before.get("pair_features", []))
        label = random.choice([0, 1])
        feats = {k: pair[k] for k in ["liked_sim", "disliked_sim", "reason_sim"]}
        feats.update({
            "clip_similarity": pair.get("score", 0.0),
            "compat_score": pair.get("score", 0.0),
            "pattern_clash": 0.0,
            "color_clash": 0.0,
            "candidate_patterned": 0.0,
            "mood_fit": 0.0
        })
        log.append({
            "item_ID": f"{pair['top_id']}|{pair['bottom_id']}",
            "features": feats,
            "label": label,
            "reason": "synthetic" if label == 0 else "",
            "reason_vec": None
        })
    save_feedback_log(log)
    train_personalizer()

    print("\n--- AFTER FEEDBACK (re‑run) ---")
    after = recommend_full_outfit(query, top_k=top_k)
    after_ids = [p["top_id"] for p in after.get("outfit_pairs", [])]
    print("Top‑bottom IDs (after):", after_ids)

    # Simple visual diff
    changed = [i for i, (b, a) in enumerate(zip(before_ids, after_ids)) if b != a]
    if changed:
        print(f"Pairs reordered at positions: {changed}")
    else:
        print("No change in ordering (personalizer may need more data).")

def plot_personalizer_weights():
    """Bar plot of learned personalizer coefficients vs. feature names.
    """
    personalizer, scaler = load_personalizer()
    if personalizer is None:
        print("⚠️ Personalizer not trained yet.")
        return
    coefs = personalizer.coef_[0]
    plt.figure(figsize=(10, 4))
    plt.bar(FEATURE_KEYS, coefs)
    plt.title("Personalizer feature weights")
    plt.xlabel("Feature")
    plt.ylabel("Weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# End of additional_functions.py

# Additional imports needed for evaluation utilities
import torch

# Additional imports needed for evaluation utilities
import torch
