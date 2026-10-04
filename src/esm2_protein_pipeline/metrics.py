"""Classification metrics and trivial baselines for ESM-2 sequence-classification adaptation.

Pure Python (no scikit-learn): accuracy, macro-F1, per-class precision/recall/F1/support, and AUROC
(binary: positive class = the last entry of `classes`; multiclass: macro one-vs-rest), computed by
the Mann-Whitney rank statistic with average ranks for ties.

Four trivial baselines frame a fine-tuned model (EVAL10/EVAL11): the majority class; one threshold on the
hydrophobic residue fraction (the one composition feature the sample generator equalises); a nearest-centroid
rule on the full 20-residue composition (order-blind); and one threshold on the longest hydrophobic run (an
order-aware one-line rule). All are fitted on the training split only (SPL8).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import STANDARD_AMINO_ACIDS
from .samples import hydrophobic_fraction, longest_hydrophobic_run


def _prf(hits: int, n_pred: int, n_true: int) -> dict[str, float]:
    precision = hits / n_pred if n_pred else 0.0
    recall = hits / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def auroc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    """Area under the ROC curve for binary 0/1 labels; None when only one class is present."""
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based average rank of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, y_true, strict=True) if y == 1)
    return round((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg), 4)


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: Sequence[Sequence[float]] | None,
    classes: Sequence[str],
) -> dict[str, Any]:
    """Discrete and ranking metrics over one evaluation split (labels are class names).

    `scores[i][k]` is the score of class `classes[k]` for record i (softmax outputs from the
    pipeline; any monotone score works for AUROC). Class order is preserved exactly as given.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels vs {len(y_pred)} predictions")
    class_list = list(classes)
    unknown = sorted((set(y_true) | set(y_pred)) - set(class_list))
    if unknown:
        raise ValueError(f"labels outside the class list {class_list}: {unknown}")
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in class_list:
        hits = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == c and p == c)
        n_pred = sum(1 for p in y_pred if p == c)
        n_true = sum(1 for t in y_true if t == c)
        prf = _prf(hits, n_pred, n_true)
        per_class[c] = {**prf, "support": n_true, "predicted": n_pred}
        if n_true:
            f1s.append(prf["f1"])
    result: dict[str, Any] = {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "classes": class_list,
        "decision_rule": "argmax over class scores",
        "auroc": None,
        "auroc_definition": None,
    }
    if scores is not None and n:
        if len(scores) != n or any(len(row) != len(class_list) for row in scores):
            raise ValueError("scores must be one row per record with one column per class")
        if len(class_list) == 2:
            pos = class_list[-1]
            result["auroc"] = auroc([1 if t == pos else 0 for t in y_true], [row[-1] for row in scores])
            result["auroc_definition"] = f"binary AUROC with positive class {pos!r} (last class in the list)"
        else:
            values = []
            for k, c in enumerate(class_list):
                a = auroc([1 if t == c else 0 for t in y_true], [row[k] for row in scores])
                if a is not None:
                    values.append(a)
            result["auroc"] = round(sum(values) / len(values), 4) if values else None
            result["auroc_definition"] = "macro-averaged one-vs-rest AUROC over classes present in the split"
    return result


def majority_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Predict the most frequent training class for every evaluation record (EVAL11)."""
    counts: dict[str, int] = {}
    for r in train_records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    majority = max(sorted(counts), key=counts.__getitem__)
    metrics = classification_metrics(
        [r["label"] for r in eval_records], [majority] * len(eval_records), None, classes
    )
    return {"baseline": "majority-class", "predicted_label": majority, **metrics}


def _threshold_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
    feature: Any,
    *,
    name: str,
    feature_name: str,
    caller: str,
    threshold_format: str = ".4f",
) -> dict[str, Any]:
    """One threshold on a scalar sequence feature, direction and cut chosen for training accuracy (binary)."""
    class_list = list(classes)
    if len(class_list) != 2:
        raise ValueError(f"{caller} is defined for binary tasks only")
    lo, hi = class_list
    train_x = [float(feature(r["sequence"])) for r in train_records]
    train_y = [r["label"] for r in train_records]
    candidates = sorted(set(train_x))
    best = (-1.0, 0.0, True)  # accuracy, threshold, high_is_hi
    for t in candidates:
        for high_is_hi in (True, False):
            pred = [(hi if (x >= t) == high_is_hi else lo) for x in train_x]
            acc = sum(p == y for p, y in zip(pred, train_y, strict=True)) / len(train_y)
            if acc > best[0]:
                best = (acc, t, high_is_hi)
    _, threshold, high_is_hi = best
    eval_x = [float(feature(r["sequence"])) for r in eval_records]
    eval_pred = [(hi if (x >= threshold) == high_is_hi else lo) for x in eval_x]
    # Score for AUROC: the feature itself, oriented so that a larger score favours `hi`.
    scores = [[1.0 - x, x] if high_is_hi else [x, 1.0 - x] for x in eval_x]
    metrics = classification_metrics([r["label"] for r in eval_records], eval_pred, scores, class_list)
    return {
        "baseline": name,
        "threshold": round(threshold, 4),
        "rule": f"predict {hi!r} when {feature_name} {'>=' if high_is_hi else '<'} "
        f"{threshold:{threshold_format}}",
        "train_accuracy": round(best[0], 4),
        **metrics,
    }


def composition_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Threshold on the hydrophobic residue fraction, fitted on the training split (binary only).

    The threshold and the class direction are chosen to maximise training accuracy; the evaluation
    split is never touched during fitting (SPL8). On the synthetic sample every `scattered` record has
    the hydrophobic count of a `segment` record, so this one feature cannot separate the classes. That
    does not make the classes equal in composition: see `residue_composition_baseline`.
    """
    return _threshold_baseline(
        train_records,
        eval_records,
        classes,
        hydrophobic_fraction,
        name="hydrophobic-fraction threshold",
        feature_name="fraction",
        caller="composition_baseline",
    )


def longest_run_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Threshold on `longest_hydrophobic_run`, fitted on the training split (binary only).

    An order-aware one-line rule: it reads where the hydrophobic residues sit, not how many there are.
    On the synthetic sample it is the rule that defines the classes, so a fine-tuned model can at best
    equal it there (review ESM-M2).
    """
    return _threshold_baseline(
        train_records,
        eval_records,
        classes,
        longest_hydrophobic_run,
        name="longest-hydrophobic-run threshold",
        feature_name="longest hydrophobic run",
        caller="longest_run_baseline",
        threshold_format="g",
    )


def residue_composition(sequence: str) -> list[float]:
    """Fractions of the 20 standard residues, in `STANDARD_AMINO_ACIDS` order (order-blind)."""
    n = len(sequence) or 1
    return [sequence.count(ch) / n for ch in STANDARD_AMINO_ACIDS]


def residue_composition_baseline(
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
) -> dict[str, Any]:
    """Nearest class centroid of the 20-residue composition vector, fitted on the training split.

    Order-blind: shuffling a sequence leaves its prediction unchanged. Any number of classes. The class
    scores are a softmax over negative Euclidean distances to the centroids (a ranking, not a probability).
    """
    class_list = list(classes)
    sums: dict[str, list[float]] = {c: [0.0] * len(STANDARD_AMINO_ACIDS) for c in class_list}
    counts: dict[str, int] = {c: 0 for c in class_list}
    for r in train_records:
        acc = sums[r["label"]]
        for k, v in enumerate(residue_composition(r["sequence"])):
            acc[k] += v
        counts[r["label"]] += 1
    empty = [c for c in class_list if not counts[c]]
    if empty:
        raise ValueError(f"classes {empty} have no training records; cannot fit a centroid")
    centroids = {c: [v / counts[c] for v in sums[c]] for c in class_list}

    def predict(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[float]]]:
        labels, scores = [], []
        for r in records:
            vec = residue_composition(r["sequence"])
            dist = [math.dist(vec, centroids[c]) for c in class_list]
            labels.append(class_list[min(range(len(class_list)), key=dist.__getitem__)])
            nearest = min(dist)
            exps = [math.exp(nearest - d) for d in dist]
            total = sum(exps)
            scores.append([e / total for e in exps])
        return labels, scores

    train_pred, _ = predict(train_records)
    hits = sum(p == r["label"] for p, r in zip(train_pred, train_records, strict=True))
    eval_pred, eval_scores = predict(eval_records)
    metrics = classification_metrics([r["label"] for r in eval_records], eval_pred, eval_scores, class_list)
    return {
        "baseline": "20-residue composition nearest centroid",
        "rule": "predict the class whose mean training composition is nearest (Euclidean)",
        "train_accuracy": round(hits / len(train_records), 4) if train_records else 0.0,
        **metrics,
    }
